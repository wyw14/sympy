"""
This module provides convenient functions to transform SymPy expressions to
lambda functions which can be used to calculate numerical values very fast.
"""

from __future__ import annotations
from typing import Any

import builtins
import importlib
import inspect
import io
import keyword
import textwrap
import linecache
import tokenize
import weakref

# Required despite static analysis claiming it is not used
from sympy.external import import_module # noqa:F401
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import (iterable,
    NotIterable, flatten)
from sympy.utilities.misc import filldedent


__doctest_requires__ = {('lambdify',): ['numpy', 'tensorflow']}


# Default namespaces, letting us define translations that can't be defined
# by simple variable maps, like I => 1j
MATH_DEFAULT: dict[str, Any] = {}
CMATH_DEFAULT: dict[str,Any] = {}
MPMATH_DEFAULT: dict[str, Any] = {}
UMATH_DEFAULT: dict[str, Any] = {}
NUMPY_DEFAULT: dict[str, Any] = {"I": 1j}
UNUMPY_DEFAULT: dict[str, Any] = {"I": 1j}
SCIPY_DEFAULT: dict[str, Any] = {"I": 1j}
CUPY_DEFAULT: dict[str, Any] = {"I": 1j}
JAX_DEFAULT: dict[str, Any] = {"I": 1j}
TENSORFLOW_DEFAULT: dict[str, Any] = {}
TORCH_DEFAULT: dict[str, Any] = {"I": 1j}
SYMPY_DEFAULT: dict[str, Any] = {}
NUMEXPR_DEFAULT: dict[str, Any] = {}

# These are the namespaces the lambda functions will use.
# These are separate from the names above because they are modified
# throughout this file, whereas the defaults should remain unmodified.

MATH = MATH_DEFAULT.copy()
CMATH = CMATH_DEFAULT.copy()
MPMATH = MPMATH_DEFAULT.copy()
UMATH = UMATH_DEFAULT.copy()
NUMPY = NUMPY_DEFAULT.copy()
UNUMPY = UNUMPY_DEFAULT.copy()
SCIPY = SCIPY_DEFAULT.copy()
CUPY = CUPY_DEFAULT.copy()
JAX = JAX_DEFAULT.copy()
TENSORFLOW = TENSORFLOW_DEFAULT.copy()
TORCH = TORCH_DEFAULT.copy()
SYMPY = SYMPY_DEFAULT.copy()
NUMEXPR = NUMEXPR_DEFAULT.copy()


# Mappings between SymPy and other modules function names.
MATH_TRANSLATIONS = {
    "ceiling": "ceil",
    "E": "e",
    "ln": "log",
}

CMATH_TRANSLATIONS: dict[str, str] = {}

# NOTE: This dictionary is reused in Function._eval_evalf to allow subclasses
# of Function to automatically evalf.
MPMATH_TRANSLATIONS = {
    "Abs": "fabs",
    "elliptic_k": "ellipk",
    "elliptic_f": "ellipf",
    "elliptic_e": "ellipe",
    "elliptic_pi": "ellippi",
    "ceiling": "ceil",
    "chebyshevt": "chebyt",
    "chebyshevu": "chebyu",
    "assoc_legendre": "legenp",
    "E": "e",
    "I": "j",
    "ln": "log",
    #"lowergamma":"lower_gamma",
    "oo": "inf",
    #"uppergamma":"upper_gamma",
    "LambertW": "lambertw",
    "MutableDenseMatrix": "matrix",
    "ImmutableDenseMatrix": "matrix",
    "conjugate": "conj",
    "dirichlet_eta": "altzeta",
    "Ei": "ei",
    "Shi": "shi",
    "Chi": "chi",
    "Si": "si",
    "Ci": "ci",
    "RisingFactorial": "rf",
    "FallingFactorial": "ff",
    "betainc_regularized": "betainc",
}

UMATH_TRANSLATIONS: dict[str, str] = {
    "ceiling": "ceil",
    "E": "e",
    "ln": "log",
}

NUMPY_TRANSLATIONS: dict[str, str] = {
    "Heaviside": "heaviside",
}

UNUMPY_TRANSLATIONS: dict[str, str] = {}

SCIPY_TRANSLATIONS: dict[str, str] = {
    "jn" : "spherical_jn",
    "yn" : "spherical_yn"
}
CUPY_TRANSLATIONS: dict[str, str] = {}
JAX_TRANSLATIONS: dict[str, str] = {}

TENSORFLOW_TRANSLATIONS: dict[str, str] = {}
TORCH_TRANSLATIONS: dict[str, str] = {}

NUMEXPR_TRANSLATIONS: dict[str, str] = {}

# Available modules:
MODULES = {
    "math": (MATH, MATH_DEFAULT, MATH_TRANSLATIONS, ("from math import *",)),
    "cmath": (CMATH, CMATH_DEFAULT, CMATH_TRANSLATIONS, ("import cmath; from cmath import *",)),
    "mpmath": (MPMATH, MPMATH_DEFAULT, MPMATH_TRANSLATIONS, ("from mpmath import *",)),
    "umath": (UMATH, UMATH_DEFAULT, UMATH_TRANSLATIONS, ("from math import pi, e, tau, inf, nan; from uncertainties.umath import *",)),
    "numpy": (NUMPY, NUMPY_DEFAULT, NUMPY_TRANSLATIONS, ("import numpy; from numpy import *; from numpy.linalg import *",)),
    "unumpy": (UNUMPY, UNUMPY_DEFAULT, UNUMPY_TRANSLATIONS, ("import uncertainties.unumpy; from uncertainties.unumpy import *; from uncertainties.unumpy.ulinalg import *",)),
    "scipy": (SCIPY, SCIPY_DEFAULT, SCIPY_TRANSLATIONS, ("import scipy; import numpy; from scipy.special import *",)),
    "cupy": (CUPY, CUPY_DEFAULT, CUPY_TRANSLATIONS, ("import cupy",)),
    "jax": (JAX, JAX_DEFAULT, JAX_TRANSLATIONS, ("import jax",)),
    "tensorflow": (TENSORFLOW, TENSORFLOW_DEFAULT, TENSORFLOW_TRANSLATIONS, ("import tensorflow",)),
    "torch": (TORCH, TORCH_DEFAULT, TORCH_TRANSLATIONS, ("import torch",)),
    "sympy": (SYMPY, SYMPY_DEFAULT, {}, (
        "from sympy.functions import *",
        "from sympy.matrices import *",
        "from sympy import Integral, pi, oo, nan, zoo, E, I",)),
    "numexpr" : (NUMEXPR, NUMEXPR_DEFAULT, NUMEXPR_TRANSLATIONS,
                 ("import_module('numexpr')", )),
}


def _import(module, reload=False):
    """
    Creates a global translation dictionary for module.

    The argument module has to be one of the following strings: "math","cmath"
    "mpmath", "umath", "numpy", "unumpy", "sympy", "tensorflow", "jax".
    These dictionaries map names of Python functions to their equivalent in
    other modules.
    """
    try:
        namespace, namespace_default, translations, import_commands = MODULES[
            module]
    except KeyError:
        raise NameError(
            "'%s' module cannot be used for lambdification" % module)

    # Clear namespace or exit
    if namespace != namespace_default:
        # The namespace was already generated, don't do it again if not forced.
        if reload:
            namespace.clear()
            namespace.update(namespace_default)
        else:
            return

    for import_command in import_commands:
        if import_command.startswith('import_module'):
            module = eval(import_command)

            if module is not None:
                namespace.update(module.__dict__)
                continue
        else:
            try:
                exec(import_command, {}, namespace)
                continue
            except ImportError:
                pass

        raise ImportError(
            "Cannot import '%s' with '%s' command" % (module, import_command))

    # Add translated names to namespace
    for sympyname, translation in translations.items():
        namespace[sympyname] = namespace[translation]

    # For computing the modulus of a SymPy expression we use the builtin abs
    # function, instead of the previously used fabs function for all
    # translation modules. This is because the fabs function in the math
    # module does not accept complex valued arguments. (see issue 9474). The
    # only exception, where we don't use the builtin abs function is the
    # mpmath translation module, because mpmath.fabs returns mpf objects in
    # contrast to abs().
    if 'Abs' not in namespace:
        namespace['Abs'] = abs

# Used for dynamically generated filenames that are inserted into the
# linecache.
_lambdify_generated_counter = 1


@doctest_depends_on(modules=('numpy', 'scipy', 'tensorflow', 'uncertainties'), python_version=(3,))
def lambdify(args, expr, modules=None, printer=None, use_imps=True,
             dummify=False, cse=False, docstring_limit=1000):
    """Convert a SymPy expression into a function that allows for fast
    numeric evaluation.

    .. warning::
       This function uses ``exec``, and thus should not be used on
       unsanitized input.

    .. deprecated:: 1.7
       Passing a set for the *args* parameter is deprecated as sets are
       unordered. Use an ordered iterable such as a list or tuple.

    Explanation
    ===========

    For example, to convert the SymPy expression ``sin(x) + cos(x)`` to an
    equivalent NumPy function that numerically evaluates it:

    >>> from sympy import sin, cos, symbols, lambdify
    >>> import numpy as np
    >>> x = symbols('x')
    >>> expr = sin(x) + cos(x)
    >>> expr
    sin(x) + cos(x)
    >>> f = lambdify(x, expr, 'numpy')
    >>> a = np.array([1, 2])
    >>> f(a)
    [1.38177329 0.49315059]

    The primary purpose of this function is to provide a bridge from SymPy
    expressions to numerical libraries such as NumPy, SciPy, NumExpr, mpmath,
    and tensorflow. In general, SymPy functions do not work with objects from
    other libraries, such as NumPy arrays, and functions from numeric
    libraries like NumPy or mpmath do not work on SymPy expressions.
    ``lambdify`` bridges the two by converting a SymPy expression to an
    equivalent numeric function.

    The basic workflow with ``lambdify`` is to first create a SymPy expression
    representing whatever mathematical function you wish to evaluate. This
    should be done using only SymPy functions and expressions. Then, use
    ``lambdify`` to convert this to an equivalent function for numerical
    evaluation. For instance, above we created ``expr`` using the SymPy symbol
    ``x`` and SymPy functions ``sin`` and ``cos``, then converted it to an
    equivalent NumPy function ``f``, and called it on a NumPy array ``a``.

    Parameters
    ==========

    args : List[Symbol]
        A variable or a list of variables whose nesting represents the
        nesting of the arguments that will be passed to the function.

        Variables can be symbols, undefined functions, or matrix symbols.

        >>> from sympy import Eq
        >>> from sympy.abc import x, y, z

        The list of variables should match the structure of how the
        arguments will be passed to the function. Simply enclose the
        parameters as they will be passed in a list.

        To call a function like ``f(x)`` then ``[x]``
        should be the first argument to ``lambdify``; for this
        case a single ``x`` can also be used:

        >>> f = lambdify(x, x + 1)
        >>> f(1)
        2
        >>> f = lambdify([x], x + 1)
        >>> f(1)
        2

        To call a function like ``f(x, y)`` then ``[x, y]`` will
        be the first argument of the ``lambdify``:

        >>> f = lambdify([x, y], x + y)
        >>> f(1, 1)
        2

        To call a function with a single 3-element tuple like
        ``f((x, y, z))`` then ``[(x, y, z)]`` will be the first
        argument of the ``lambdify``:

        >>> f = lambdify([(x, y, z)], Eq(z**2, x**2 + y**2))
        >>> f((3, 4, 5))
        True

        If two args will be passed and the first is a scalar but
        the second is a tuple with two arguments then the items
        in the list should match that structure:

        >>> f = lambdify([x, (y, z)], x + y + z)
        >>> f(1, (2, 3))
        6

    expr : Expr
        An expression, list of expressions, or matrix to be evaluated.

        Lists may be nested.
        If the expression is a list, the output will also be a list.

        >>> f = lambdify(x, [x, [x + 1, x + 2]])
        >>> f(1)
        [1, [2, 3]]

        If it is a matrix, an array will be returned (for the NumPy module).

        >>> from sympy import Matrix
        >>> f = lambdify(x, Matrix([x, x + 1]))
        >>> f(1)
        [[1]
        [2]]

        Note that the argument order here (variables then expression) is used
        to emulate the Python ``lambda`` keyword. ``lambdify(x, expr)`` works
        (roughly) like ``lambda x: expr``
        (see :ref:`lambdify-how-it-works` below).

    modules : str, optional
        Specifies the numeric library to use.

        If not specified, *modules* defaults to:

        - ``["scipy", "numpy"]`` if SciPy is installed
        - ``["numpy"]`` if only NumPy is installed
        - ``["math","cmath", "mpmath", "sympy"]`` if neither is installed.

        That is, SymPy functions are replaced as far as possible by
        either ``scipy`` or ``numpy`` functions if available, and Python's
        standard library ``math`` and ``cmath``, or ``mpmath`` functions otherwise.

        *modules* can be one of the following types:

        - The strings ``"math"``, ``"cmath"``, ``"mpmath"``, ``"umath"``,
          ``"numpy"``, ``"unumpy"``, ``"numexpr"``, ``"scipy"``, ``"sympy"``,
          ``"tensorflow"``, ``"torch"`` or ``"jax"``. This uses the corresponding printer
          and namespace mapping for that module.
        - A module (e.g., ``math``). This uses the global namespace of the
          module. If the module is one of the above known modules, it will
          also use the corresponding printer and namespace mapping
          (i.e., ``modules=numpy`` is equivalent to ``modules="numpy"``).
        - A dictionary that maps names of SymPy functions to arbitrary
          functions
          (e.g., ``{'sin': custom_sin}``).
        - A list that contains a mix of the arguments above, with higher
          priority given to entries appearing first
          (e.g., to use the NumPy module but override the ``sin`` function
          with a custom version, you can use
          ``[{'sin': custom_sin}, 'numpy']``).

    dummify : bool, optional
        Whether or not the variables in the provided expression that are not
        valid Python identifiers are substituted with dummy symbols.

        This allows for undefined functions like ``Function('f')(t)`` to be
        supplied as arguments. By default, the variables are only dummified
        if they are not valid Python identifiers.

        Set ``dummify=True`` to replace all arguments with dummy symbols
        (if ``args`` is not a string) - for example, to ensure that the
        arguments do not redefine any built-in names.

    cse : bool, or callable, optional
        Large expressions can be computed more efficiently when
        common subexpressions are identified and precomputed before
        being used multiple time. Finding the subexpressions will make
        creation of the 'lambdify' function slower, however.

        When ``True``, ``sympy.simplify.cse`` is used, otherwise (the default)
        the user may pass a function matching the ``cse`` signature.

    docstring_limit : int or None
        When lambdifying large expressions, a significant proportion of the time
        spent inside ``lambdify`` is spent producing a string representation of
        the expression for use in the automatically generated docstring of the
        returned function. For expressions containing hundreds or more nodes the
        resulting docstring often becomes so long and dense that it is difficult
        to read. To reduce the runtime of lambdify, the rendering of the full
        expression inside the docstring can be disabled.

        When ``None``, the full expression is rendered in the docstring. When
        ``0`` or a negative ``int``, an ellipsis is rendering in the docstring
        instead of the expression. When a strictly positive ``int``, if the
        number of nodes in the expression exceeds ``docstring_limit`` an
        ellipsis is rendered in the docstring, otherwise a string representation
        of the expression is rendered as normal. The default is ``1000``.

    Examples
    ========

    >>> from sympy.utilities.lambdify import implemented_function
    >>> from sympy import sqrt, sin, Matrix
    >>> from sympy import Function
    >>> from sympy.abc import w, x, y, z

    >>> f = lambdify(x, x**2)
    >>> f(2)
    4
    >>> f = lambdify((x, y, z), [z, y, x])
    >>> f(1,2,3)
    [3, 2, 1]
    >>> f = lambdify(x, sqrt(x))
    >>> f(4)
    2.0
    >>> f = lambdify((x, y), sin(x*y)**2)
    >>> f(0, 5)
    0.0
    >>> row = lambdify((x, y), Matrix((x, x + y)).T, modules='sympy')
    >>> row(1, 2)
    Matrix([[1, 3]])

    ``lambdify`` can be used to translate SymPy expressions into mpmath
    functions. This may be preferable to using ``evalf`` (which uses mpmath on
    the backend) in some cases.

    >>> f = lambdify(x, sin(x), 'mpmath')
    >>> f(1)
    0.8414709848078965

    Tuple arguments are handled and the lambdified function should
    be called with the same type of arguments as were used to create
    the function:

    >>> f = lambdify((x, (y, z)), x + y)
    >>> f(1, (2, 4))
    3

    The ``flatten`` function can be used to always work with flattened
    arguments:

    >>> from sympy.utilities.iterables import flatten
    >>> args = w, (x, (y, z))
    >>> vals = 1, (2, (3, 4))
    >>> f = lambdify(flatten(args), w + x + y + z)
    >>> f(*flatten(vals))
    10

    Functions present in ``expr`` can also carry their own numerical
    implementations, in a callable attached to the ``_imp_`` attribute. This
    can be used with undefined functions using the ``implemented_function``
    factory:

    >>> f = implemented_function(Function('f'), lambda x: x+1)
    >>> func = lambdify(x, f(x))
    >>> func(4)
    5

    ``lambdify`` always prefers ``_imp_`` implementations to implementations
    in other namespaces, unless the ``use_imps`` input parameter is False.

    Usage with Tensorflow:

    >>> import tensorflow as tf
    >>> from sympy import Max, sin, lambdify
    >>> from sympy.abc import x

    >>> f = Max(x, sin(x))
    >>> func = lambdify(x, f, 'tensorflow')

    After tensorflow v2, eager execution is enabled by default.
    If you want to get the compatible result across tensorflow v1 and v2
    as same as this tutorial, run this line.

    >>> tf.compat.v1.enable_eager_execution()

    If you have eager execution enabled, you can get the result out
    immediately as you can use numpy.

    If you pass tensorflow objects, you may get an ``EagerTensor``
    object instead of value.

    >>> result = func(tf.constant(1.0))
    >>> print(result)
    tf.Tensor(1.0, shape=(), dtype=float32)
    >>> print(result.__class__)
    <class 'tensorflow.python.framework.ops.EagerTensor'>

    You can use ``.numpy()`` to get the numpy value of the tensor.

    >>> result.numpy()
    1.0

    >>> var = tf.Variable(2.0)
    >>> result = func(var) # also works for tf.Variable and tf.Placeholder
    >>> result.numpy()
    2.0

    And it works with any shape array.

    >>> tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    >>> result = func(tensor)
    >>> result.numpy()
    [[1. 2.]
     [3. 4.]]

    Usage with the Uncertainties package:

    >>> from uncertainties import ufloat
    >>> from sympy.abc import x

    Use with scalar values by passing ``modules='umath'``.

    >>> uncert_square = lambdify(x, x**2, 'umath')
    >>> uncert_square(ufloat(2, 0.1))
    4.00+/-0.40
    >>> uncert_square(2) # Also works with non ``ufloat`` inputs
    4

    Use with vectors and matrices by passing ``modules='unumpy'``.

    >>> uncert_mat_square = lambdify([x, y], Matrix([x, y]).dot(Matrix([2, 3])), 'unumpy')
    >>> uncert_mat_square(ufloat(1, 0.1), ufloat(2, 0.2))
     8.00+/-0.6

    Due to the internal workings of ``uncertainties.unumpy``, when a
    matrix is returned, some elements may unexpectedly be wrapped in a
    1x1 array. This can be fixed by applying an identity
    operation to the result, such as multiplying by 1 or adding 0.

    >>> from sympy import sqrt
    >>> uncert_sqrt = lambdify(x, Matrix([sqrt(x)]), 'unumpy')
    >>> uncert_sqrt(ufloat(4, 0.4)) # Matrix contains element wrapped in 1x1 array.
    [[array(2.0+/-0.1, dtype=object)]]
    >>> uncert_sqrt(ufloat(4, 0.4)) * 1 # Fix by applying identity operation on result
    [[2.0+/-0.1]]

    Notes
    =====

    - For functions involving large array calculations, numexpr can provide a
      significant speedup over numpy. Please note that the available functions
      for numexpr are more limited than numpy but can be expanded with
      ``implemented_function`` and user defined subclasses of Function. If
      specified, numexpr may be the only option in modules. The official list
      of numexpr functions can be found at:
      https://numexpr.readthedocs.io/en/latest/user_guide.html#supported-functions

    - In the above examples, the generated functions can accept scalar
      values or numpy arrays as arguments.  However, in some cases
      the generated function relies on the input being a numpy array:

      >>> import numpy
      >>> from sympy import Piecewise
      >>> from sympy.testing.pytest import ignore_warnings
      >>> f = lambdify(x, Piecewise((x, x <= 1), (1/x, x > 1)), "numpy")

      >>> with ignore_warnings(RuntimeWarning):
      ...     f(numpy.array([-1, 0, 1, 2]))
      [-1.   0.   1.   0.5]

      >>> f(0)
      Traceback (most recent call last):
          ...
      ZeroDivisionError: division by zero

      In such cases, the input should be wrapped in a numpy array:

      >>> with ignore_warnings(RuntimeWarning):
      ...     float(f(numpy.array([0])))
      0.0

      Or if numpy functionality is not required another module can be used:

      >>> f = lambdify(x, Piecewise((x, x <= 1), (1/x, x > 1)), "math")
      >>> f(0)
      0

    .. _lambdify-how-it-works:

    How it works
    ============

    When using this function, it helps a great deal to have an idea of what it
    is doing. At its core, lambdify is nothing more than a namespace
    translation, on top of a special printer that makes some corner cases work
    properly.

    To understand lambdify, first we must properly understand how Python
    namespaces work. Say we had two files. One called ``sin_cos_sympy.py``,
    with

    .. code:: python

        # sin_cos_sympy.py

        from sympy.functions.elementary.trigonometric import (cos, sin)

        def sin_cos(x):
            return sin(x) + cos(x)


    and one called ``sin_cos_numpy.py`` with

    .. code:: python

        # sin_cos_numpy.py

        from numpy import sin, cos

        def sin_cos(x):
            return sin(x) + cos(x)

    The two files define an identical function ``sin_cos``. However, in the
    first file, ``sin`` and ``cos`` are defined as the SymPy ``sin`` and
    ``cos``. In the second, they are defined as the NumPy versions.

    If we were to import the first file and use the ``sin_cos`` function, we
    would get something like

    >>> from sin_cos_sympy import sin_cos # doctest: +SKIP
    >>> sin_cos(1) # doctest: +SKIP
    cos(1) + sin(1)

    On the other hand, if we imported ``sin_cos`` from the second file, we
    would get

    >>> from sin_cos_numpy import sin_cos # doctest: +SKIP
    >>> sin_cos(1) # doctest: +SKIP
    1.38177329068

    In the first case we got a symbolic output, because it used the symbolic
    ``sin`` and ``cos`` functions from SymPy. In the second, we got a numeric
    result, because ``sin_cos`` used the numeric ``sin`` and ``cos`` functions
    from NumPy. But notice that the versions of ``sin`` and ``cos`` that were
    used was not inherent to the ``sin_cos`` function definition. Both
    ``sin_cos`` definitions are exactly the same. Rather, it was based on the
    names defined at the module where the ``sin_cos`` function was defined.

    The key point here is that when function in Python references a name that
    is not defined in the function, that name is looked up in the "global"
    namespace of the module where that function is defined.

    Now, in Python, we can emulate this behavior without actually writing a
    file to disk using the ``exec`` function. ``exec`` takes a string
    containing a block of Python code, and a dictionary that should contain
    the global variables of the module. It then executes the code "in" that
    dictionary, as if it were the module globals. The following is equivalent
    to the ``sin_cos`` defined in ``sin_cos_sympy.py``:

    >>> import sympy
    >>> module_dictionary = {'sin': sympy.sin, 'cos': sympy.cos}
    >>> exec('''
    ... def sin_cos(x):
    ...     return sin(x) + cos(x)
    ... ''', module_dictionary)
    >>> sin_cos = module_dictionary['sin_cos']
    >>> sin_cos(1)
    cos(1) + sin(1)

    and similarly with ``sin_cos_numpy``:

    >>> import numpy
    >>> module_dictionary = {'sin': numpy.sin, 'cos': numpy.cos}
    >>> exec('''
    ... def sin_cos(x):
    ...     return sin(x) + cos(x)
    ... ''', module_dictionary)
    >>> sin_cos = module_dictionary['sin_cos']
    >>> sin_cos(1)
    1.38177329068

    So now we can get an idea of how ``lambdify`` works. The name "lambdify"
    comes from the fact that we can think of something like ``lambdify(x,
    sin(x) + cos(x), 'numpy')`` as ``lambda x: sin(x) + cos(x)``, where
    ``sin`` and ``cos`` come from the ``numpy`` namespace. This is also why
    the symbols argument is first in ``lambdify``, as opposed to most SymPy
    functions where it comes after the expression: to better mimic the
    ``lambda`` keyword.

    ``lambdify`` takes the input expression (like ``sin(x) + cos(x)``) and

    1. Converts it to a string
    2. Creates a module globals dictionary based on the modules that are
       passed in (by default, it uses the NumPy module)
    3. Creates the string ``"def func({vars}): return {expr}"``, where ``{vars}`` is the
       list of variables separated by commas, and ``{expr}`` is the string
       created in step 1., then ``exec``s that string with the module globals
       namespace and returns ``func``.

    In fact, functions returned by ``lambdify`` support inspection. So you can
    see exactly how they are defined by using ``inspect.getsource``, or ``??`` if you
    are using IPython or the Jupyter notebook.

    >>> f = lambdify(x, sin(x) + cos(x))
    >>> import inspect
    >>> print(inspect.getsource(f))
    def _lambdifygenerated(x):
        return sin(x) + cos(x)

    This shows us the source code of the function, but not the namespace it
    was defined in. We can inspect that by looking at the ``__globals__``
    attribute of ``f``:

    >>> f.__globals__['sin']
    <ufunc 'sin'>
    >>> f.__globals__['cos']
    <ufunc 'cos'>
    >>> f.__globals__['sin'] is numpy.sin
    True

    This shows us that ``sin`` and ``cos`` in the namespace of ``f`` will be
    ``numpy.sin`` and ``numpy.cos``.

    Note that there are some convenience layers in each of these steps, but at
    the core, this is how ``lambdify`` works. Step 1 is done using the
    ``LambdaPrinter`` printers defined in the printing module (see
    :mod:`sympy.printing.lambdarepr`). This allows different SymPy expressions
    to define how they should be converted to a string for different modules.
    You can change which printer ``lambdify`` uses by passing a custom printer
    in to the ``printer`` argument.

    Step 2 is augmented by certain translations. There are default
    translations for each module, but you can provide your own by passing a
    list to the ``modules`` argument. For instance,

    >>> def mysin(x):
    ...     print('taking the sin of', x)
    ...     return numpy.sin(x)
    ...
    >>> f = lambdify(x, sin(x), [{'sin': mysin}, 'numpy'])
    >>> f(1)
    taking the sin of 1
    0.8414709848078965

    The globals dictionary is generated from the list by merging the
    dictionary ``{'sin': mysin}`` and the module dictionary for NumPy. The
    merging is done so that earlier items take precedence, which is why
    ``mysin`` is used above instead of ``numpy.sin``.

    If you want to modify the way ``lambdify`` works for a given function, it
    is usually easiest to do so by modifying the globals dictionary as such.
    In more complicated cases, it may be necessary to create and pass in a
    custom printer.

    Finally, step 3 is augmented with certain convenience operations, such as
    the addition of a docstring.

    Understanding how ``lambdify`` works can make it easier to avoid certain
    gotchas when using it. For instance, a common mistake is to create a
    lambdified function for one module (say, NumPy), and pass it objects from
    another (say, a SymPy expression).

    For instance, say we create

    >>> from sympy.abc import x
    >>> f = lambdify(x, x + 1, 'numpy')

    Now if we pass in a NumPy array, we get that array plus 1

    >>> import numpy
    >>> a = numpy.array([1, 2])
    >>> f(a)
    [2 3]

    But what happens if you make the mistake of passing in a SymPy expression
    instead of a NumPy array:

    >>> f(x + 1)
    x + 2

    This worked, but it was only by accident. Now take a different lambdified
    function:

    >>> from sympy import sin
    >>> g = lambdify(x, x + sin(x), 'numpy')

    This works as expected on NumPy arrays:

    >>> g(a)
    [1.84147098 2.90929743]

    But if we try to pass in a SymPy expression, it fails

    >>> g(x + 1)
    Traceback (most recent call last):
    ...
    TypeError: loop of ufunc does not support argument 0 of type Add which has
               no callable sin method

    Now, let's look at what happened. The reason this fails is that ``g``
    calls ``numpy.sin`` on the input expression, and ``numpy.sin`` does not
    know how to operate on a SymPy object. **As a general rule, NumPy
    functions do not know how to operate on SymPy expressions, and SymPy
    functions do not know how to operate on NumPy arrays. This is why lambdify
    exists: to provide a bridge between SymPy and NumPy.**

    However, why is it that ``f`` did work? That's because ``f`` does not call
    any functions, it only adds 1. So the resulting function that is created,
    ``def _lambdifygenerated(x): return x + 1`` does not depend on the globals
    namespace it is defined in. Thus it works, but only by accident. A future
    version of ``lambdify`` may remove this behavior.

    Be aware that certain implementation details described here may change in
    future versions of SymPy. The API of passing in custom modules and
    printers will not change, but the details of how a lambda function is
    created may change. However, the basic idea will remain the same, and
    understanding it will be helpful to understanding the behavior of
    lambdify.

    **In general: you should create lambdified functions for one module (say,
    NumPy), and only pass it input types that are compatible with that module
    (say, NumPy arrays).** Remember that by default, if the ``module``
    argument is not provided, ``lambdify`` creates functions using the NumPy
    and SciPy namespaces.
    """
    from sympy.core.symbol import Symbol
    from sympy.core.expr import Expr

    # If the user hasn't specified any modules, use what is available.
    if modules is None:
        modules = _default_modules()

    # Get the needed namespaces.
    namespaces = _collect_namespaces(expr, modules, use_imps)
    # fill namespace with first having highest priority
    namespace = _merge_namespaces(namespaces)

    if hasattr(expr, "atoms"):
        #Try if you can extract symbols from the expression.
        #Move on if expr.atoms in not implemented.
        syms = expr.atoms(Symbol)
        for term in syms:
            namespace.update({str(term): term})

    if printer is None:
        Printer = _select_printer_class(namespaces)
        user_functions = _user_functions_from_namespaces(namespaces)
        printer = Printer({'fully_qualified_modules': False, 'inline': True,
                           'allow_unknown_functions': True,
                           'user_functions': user_functions})

    if isinstance(args, set):
        sympy_deprecation_warning(
            """
Passing the function arguments to lambdify() as a set is deprecated. This
leads to unpredictable results since sets are unordered. Instead, use a list
or tuple for the function arguments.
            """,
            deprecated_since_version="1.6.3",
            active_deprecations_target="deprecated-lambdify-arguments-set",
                )

    # Get the names of the args, for creating a docstring
    iterable_args = (args,) if isinstance(args, Expr) else args
    names = []

    # Grab the callers frame, for getting the names by inspection (if needed)
    callers_local_vars = inspect.currentframe().f_back.f_locals.items() # type: ignore
    for n, var in enumerate(iterable_args):
        if hasattr(var, 'name'):
            names.append(var.name)
        else:
            # It's an iterable. Try to get name by inspection of calling frame.
            name_list = [var_name for var_name, var_val in callers_local_vars
                    if var_val is var]
            if len(name_list) == 1:
                names.append(name_list[0])
            else:
                # Cannot infer name with certainty. arg_# will have to do.
                names.append('arg_' + str(n))

    # Create the function definition code and execute it
    funcname = '_lambdifygenerated'
    if _module_present('tensorflow', namespaces):
        funcprinter = _TensorflowEvaluatorPrinter(printer, dummify)
    else:
        funcprinter = _EvaluatorPrinter(printer, dummify)

    if cse == True:
        from sympy.simplify.cse_main import cse as _cse
        cses, _expr = _cse(expr, list=False)
    elif callable(cse):
        cses, _expr = cse(expr)
    else:
        cses, _expr = (), expr
    funcstr = funcprinter.doprint(funcname, iterable_args, _expr, cses=cses)

    # Collect the module imports from the code printers.
    imp_mod_lines = []
    for mod, keys in (getattr(printer, 'module_imports', None) or {}).items():
        for k in keys:
            if k not in namespace:
                ln = "from %s import %s" % (mod, k)
                try:
                    exec(ln, {}, namespace)
                except ImportError:
                    # Tensorflow 2.0 has issues with importing a specific
                    # function from its submodule.
                    # https://github.com/tensorflow/tensorflow/issues/33022
                    ln = "%s = %s.%s" % (k, mod, k)
                    exec(ln, {}, namespace)
                imp_mod_lines.append(ln)

    # Provide lambda expression with builtins, and compatible implementation of range
    namespace.update({'builtins':builtins, 'range':range})

    funclocals = {}
    global _lambdify_generated_counter
    filename = '<lambdifygenerated-%s>' % _lambdify_generated_counter
    _lambdify_generated_counter += 1
    c = compile(funcstr, filename, 'exec')
    exec(c, namespace, funclocals)
    # mtime has to be None or else linecache.checkcache will remove it
    linecache.cache[filename] = (len(funcstr), None, funcstr.splitlines(True), filename) # type: ignore

    # Remove the entry from the linecache when the object is garbage collected
    def cleanup_linecache(filename):
        def _cleanup():
            if filename in linecache.cache:
                del linecache.cache[filename]
        return _cleanup

    func = funclocals[funcname]

    weakref.finalize(func, cleanup_linecache(filename))

    # Apply the docstring
    sig = "func({})".format(", ".join(str(i) for i in names))
    sig = textwrap.fill(sig, subsequent_indent=' '*8)
    if _too_large_for_docstring(expr, docstring_limit):
        expr_str = "EXPRESSION REDACTED DUE TO LENGTH, (see lambdify's `docstring_limit`)"
        src_str = "SOURCE CODE REDACTED DUE TO LENGTH, (see lambdify's `docstring_limit`)"
    else:
        expr_str = str(expr)
        if len(expr_str) > 78:
            expr_str = textwrap.wrap(expr_str, 75)[0] + '...'
        src_str = funcstr
    func.__doc__ = (
        "Created with lambdify. Signature:\n\n"
        "{sig}\n\n"
        "Expression:\n\n"
        "{expr}\n\n"
        "Source code:\n\n"
        "{src}\n\n"
        "Imported modules:\n\n"
        "{imp_mods}"
        ).format(sig=sig, expr=expr_str, src=src_str, imp_mods='\n'.join(imp_mod_lines))
    return func

def _module_present(modname, modlist):
    if modname in modlist:
        return True
    for m in modlist:
        if hasattr(m, '__name__') and m.__name__ == modname:
            return True
    return False

def _get_namespace(m):
    """
    This is used by _lambdify to parse its arguments.
    """
    if isinstance(m, str):
        _import(m)
        return MODULES[m][0]
    elif isinstance(m, dict):
        return m
    elif hasattr(m, "__dict__"):
        return m.__dict__
    else:
        raise TypeError("Argument must be either a string, dict or module but it is: %s" % m)


def _default_modules():
    """Select the default ``modules`` list used when ``modules`` is not given."""
    try:
        _import("scipy")
    except ImportError:
        try:
            _import("numpy")
        except ImportError:
            # Use either numpy (if available) or python.math where possible.
            # XXX: This leads to different behaviour on different systems and
            #      might be the reason for irreproducible errors.
            return ["math", "mpmath", "sympy"]
        else:
            return ["numpy"]
    else:
        return ["numpy", "scipy"]


def _collect_namespaces(expr, modules, use_imps):
    """Build the prioritized list of namespace specs used by ``lambdify``.

    Entries appearing earlier in the returned list have higher priority. The
    implementations attached to functions through ``_imp_`` end up first when
    *use_imps* is true.
    """
    namespaces = []
    # First find any function implementations
    if use_imps:
        namespaces.append(_imp_namespace(expr))
    # Check for dict before iterating
    if isinstance(modules, (dict, str)) or not hasattr(modules, '__iter__'):
        namespaces.append(modules)
    else:
        # consistency check
        if _module_present('numexpr', modules) and len(modules) > 1:
            raise TypeError("numexpr must be the only item in 'modules'")
        namespaces += list(modules)
    return namespaces


def _merge_namespaces(namespaces):
    """Merge the namespace specs, with the first one having highest priority."""
    namespace = {}
    for m in namespaces[::-1]:
        namespace.update(_get_namespace(m))
    return namespace


def _select_printer_class(namespaces):
    """Select the code printer class matching the (highest priority) module."""
    if _module_present('mpmath', namespaces):
        from sympy.printing.pycode import MpmathPrinter as Printer # type: ignore
    elif _module_present('scipy', namespaces):
        from sympy.printing.numpy import SciPyPrinter as Printer # type: ignore
    elif _module_present('numpy', namespaces):
        from sympy.printing.numpy import NumPyPrinter as Printer # type: ignore
    elif _module_present('unumpy', namespaces):
        from sympy.printing.numpy import NumPyPrinter as Printer # type: ignore
    elif _module_present('cupy', namespaces):
        from sympy.printing.numpy import CuPyPrinter as Printer # type: ignore
    elif _module_present('jax', namespaces):
        from sympy.printing.numpy import JaxPrinter as Printer # type: ignore
    elif _module_present('numexpr', namespaces):
        from sympy.printing.lambdarepr import NumExprPrinter as Printer # type: ignore
    elif _module_present('tensorflow', namespaces):
        from sympy.printing.tensorflow import TensorflowPrinter as Printer # type: ignore
    elif _module_present('torch', namespaces):
        from sympy.printing.pytorch import TorchPrinter as Printer  # type: ignore
    elif _module_present('sympy', namespaces):
        from sympy.printing.pycode import SymPyPrinter as Printer # type: ignore
    elif _module_present('cmath', namespaces):
        from sympy.printing.pycode import CmathPrinter as Printer # type: ignore
    else:
        from sympy.printing.pycode import PythonCodePrinter as Printer # type: ignore
    return Printer


def _user_functions_from_namespaces(namespaces):
    """Collect the user defined function names passed in as dictionaries."""
    user_functions = {}
    for m in namespaces[::-1]:
        if isinstance(m, dict):
            for k in m:
                user_functions[k] = k
    return user_functions



def _recursive_to_string(doprint, arg):
    """Functions in lambdify accept both SymPy types and non-SymPy types such as python
    lists and tuples. This method ensures that we only call the doprint method of the
    printer with SymPy types (so that the printer safely can use SymPy-methods)."""
    from sympy.matrices.matrixbase import MatrixBase
    from sympy.core.basic import Basic

    if isinstance(arg, (Basic, MatrixBase)):
        return doprint(arg)
    elif iterable(arg):
        if isinstance(arg, list):
            left, right = "[", "]"
        elif isinstance(arg, tuple):
            left, right = "(", ",)"
            if not arg:
                return "()"
        else:
            raise NotImplementedError("unhandled type: %s, %s" % (type(arg), arg))
        return left +', '.join(_recursive_to_string(doprint, e) for e in arg) + right
    elif isinstance(arg, str):
        return arg
    else:
        return doprint(arg)


def lambdastr(args, expr, printer=None, dummify=None):
    """
    Returns a string that can be evaluated to a lambda function.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.utilities.lambdify import lambdastr
    >>> lambdastr(x, x**2)
    'lambda x: (x**2)'
    >>> lambdastr((x,y,z), [z,y,x])
    'lambda x,y,z: ([z, y, x])'

    Although tuples may not appear as arguments to lambda in Python 3,
    lambdastr will create a lambda function that will unpack the original
    arguments so that nested arguments can be handled:

    >>> lambdastr((x, (y, z)), x + y)
    'lambda _0,_1: (lambda x,y,z: (x + y))(_0,_1[0],_1[1])'
    """
    # Transforming everything to strings.
    from sympy.matrices import DeferredVector
    from sympy.core.basic import Basic
    from sympy.core.function import (Derivative, Function)
    from sympy.core.symbol import (Dummy, Symbol)
    from sympy.core.sympify import sympify

    if printer is not None:
        if inspect.isfunction(printer):
            lambdarepr = printer
        else:
            if inspect.isclass(printer):
                lambdarepr = lambda expr: printer().doprint(expr)
            else:
                lambdarepr = lambda expr: printer.doprint(expr)
    else:
        #XXX: This has to be done here because of circular imports
        from sympy.printing.lambdarepr import lambdarepr

    def sub_args(args, dummies_dict):
        if isinstance(args, str):
            return args
        elif isinstance(args, DeferredVector):
            return str(args)
        elif iterable(args):
            dummies = flatten([sub_args(a, dummies_dict) for a in args])
            return ",".join(str(a) for a in dummies)
        else:
            # replace these with Dummy symbols
            if isinstance(args, (Function, Symbol, Derivative)):
                dummies = Dummy()
                dummies_dict.update({args : dummies})
                return str(dummies)
            else:
                return str(args)

    def sub_expr(expr, dummies_dict):
        expr = sympify(expr)
        # dict/tuple are sympified to Basic
        if isinstance(expr, Basic):
            expr = expr.xreplace(dummies_dict)
        # list is not sympified to Basic
        elif isinstance(expr, list):
            expr = [sub_expr(a, dummies_dict) for a in expr]
        return expr

    # Transform args
    def isiter(l):
        return iterable(l, exclude=(str, DeferredVector, NotIterable))

    def flat_indexes(iterable):
        for n, el in enumerate(iterable):
            if isiter(el):
                for ndeep in flat_indexes(el):
                    yield (n,) + ndeep
            else:
                yield (n,)

    if dummify is None:
        dummify = any(isinstance(a, Basic) and
            a.atoms(Function, Derivative) for a in (
            args if isiter(args) else [args]))

    if isiter(args) and any(isiter(i) for i in args):
        dum_args = [str(Dummy(str(i))) for i in range(len(args))]

        indexed_args = ','.join([
            dum_args[ind[0]] + ''.join(["[%s]" % k for k in ind[1:]])
                    for ind in flat_indexes(args)])

        lstr = lambdastr(flatten(args), expr, printer=printer, dummify=dummify)

        return 'lambda %s: (%s)(%s)' % (','.join(dum_args), lstr, indexed_args)

    dummies_dict = {}
    if dummify:
        args = sub_args(args, dummies_dict)
    else:
        if isinstance(args, str):
            pass
        elif iterable(args, exclude=DeferredVector):
            args = ",".join(str(a) for a in args)

    # Transform expr
    if dummify:
        if isinstance(expr, str):
            pass
        else:
            expr = sub_expr(expr, dummies_dict)
    expr = _recursive_to_string(lambdarepr, expr)
    return "lambda %s: (%s)" % (args, expr)

class _EvaluatorPrinter:
    def __init__(self, printer=None, dummify=False):
        self._dummify = dummify

        #XXX: This has to be done here because of circular imports
        from sympy.printing.lambdarepr import LambdaPrinter

        if printer is None:
            printer = LambdaPrinter()

        if inspect.isfunction(printer):
            self._exprrepr = printer
        else:
            if inspect.isclass(printer):
                printer = printer()

            self._exprrepr = printer.doprint

            #if hasattr(printer, '_print_Symbol'):
            #    symbolrepr = printer._print_Symbol

            #if hasattr(printer, '_print_Dummy'):
            #    dummyrepr = printer._print_Dummy

        # Used to print the generated function arguments in a standard way
        self._argrepr = LambdaPrinter().doprint

    def doprint(self, funcname, args, expr, *, cses=()):
        """
        Returns the function definition code as a string.
        """
        from sympy.core.symbol import Dummy

        funcbody = []

        if not iterable(args):
            args = [args]

        if cses:
            cses = list(cses)
            subvars, subexprs = zip(*cses)
            exprs = [expr] + list(subexprs)
            argstrs, exprs = self._preprocess(args, exprs, cses=cses)
            expr, subexprs = exprs[0], exprs[1:]
            cses = zip(subvars, subexprs)
        else:
            argstrs, expr = self._preprocess(args, expr)

        # Generate argument unpacking and final argument list
        funcargs = []
        unpackings = []

        for argstr in argstrs:
            if iterable(argstr):
                funcargs.append(self._argrepr(Dummy()))
                unpackings.extend(self._print_unpacking(argstr, funcargs[-1]))
            else:
                funcargs.append(argstr)

        funcsig = 'def {}({}):'.format(funcname, ', '.join(funcargs))

        # Wrap input arguments before unpacking
        funcbody.extend(self._print_funcargwrapping(funcargs))

        funcbody.extend(unpackings)

        for s, e in cses:
            if e is None:
                funcbody.append('del {}'.format(self._exprrepr(s)))
            else:
                funcbody.append('{} = {}'.format(self._exprrepr(s), self._exprrepr(e)))

        # Subs may appear in expressions generated by .diff()
        subs_assignments = []
        expr = self._handle_Subs(expr, out=subs_assignments)
        for lhs, rhs in subs_assignments:
            funcbody.append('{} = {}'.format(self._exprrepr(lhs), self._exprrepr(rhs)))

        str_expr = _recursive_to_string(self._exprrepr, expr)

        if '\n' in str_expr:
            str_expr = '({})'.format(str_expr)
        funcbody.append('return {}'.format(str_expr))

        funclines = [funcsig]
        funclines.extend(['    ' + line for line in funcbody])

        return '\n'.join(funclines) + '\n'

    @classmethod
    def _is_safe_ident(cls, ident):
        return isinstance(ident, str) and ident.isidentifier() \
                and not keyword.iskeyword(ident)

    def _preprocess(self, args, expr, cses=(), _dummies_dict=None):
        """Preprocess args, expr to replace arguments that do not map
        to valid Python identifiers.

        Returns string form of args, and updated expr.
        """
        from sympy.core.basic import Basic
        from sympy.core.sorting import ordered
        from sympy.core.function import (Derivative, Function)
        from sympy.core.symbol import Dummy, uniquely_named_symbol
        from sympy.matrices import DeferredVector
        from sympy.core.expr import Expr

        # Args of type Dummy can cause name collisions with args
        # of type Symbol.  Force dummify of everything in this
        # situation.
        dummify = self._dummify or any(
            isinstance(arg, Dummy) for arg in flatten(args))

        argstrs = [None]*len(args)
        if _dummies_dict is None:
            _dummies_dict = {}

        def update_dummies(arg, dummy):
            _dummies_dict[arg] = dummy
            for repl, sub in cses:
                arg = arg.xreplace({sub: repl})
                _dummies_dict[arg] = dummy

        for arg, i in reversed(list(ordered(zip(args, range(len(args)))))):
            if iterable(arg):
                s, expr = self._preprocess(arg, expr, cses=cses, _dummies_dict=_dummies_dict)
            elif isinstance(arg, DeferredVector):
                s = str(arg)
            elif isinstance(arg, Basic) and arg.is_symbol:
                s = str(arg)
                if dummify or not self._is_safe_ident(s):
                    dummy = Dummy()
                    if isinstance(expr, Expr):
                        dummy = uniquely_named_symbol(
                            dummy.name, expr, modify=lambda s: '_' + s)
                    s = self._argrepr(dummy)
                    update_dummies(arg, dummy)
                    expr = self._subexpr(expr, _dummies_dict)
            elif dummify or isinstance(arg, (Function, Derivative)):
                dummy = Dummy()
                s = self._argrepr(dummy)
                update_dummies(arg, dummy)
                expr = self._subexpr(expr, _dummies_dict)
            else:
                s = str(arg)
            argstrs[i] = s
        return argstrs, expr

    def _subexpr(self, expr, dummies_dict):
        from sympy.matrices import DeferredVector
        from sympy.core.sympify import sympify

        expr = sympify(expr)
        xreplace = getattr(expr, 'xreplace', None)
        if xreplace is not None:
            expr = xreplace(dummies_dict)
        else:
            if isinstance(expr, DeferredVector):
                pass
            elif isinstance(expr, dict):
                k = [self._subexpr(sympify(a), dummies_dict) for a in expr.keys()]
                v = [self._subexpr(sympify(a), dummies_dict) for a in expr.values()]
                expr = dict(zip(k, v))
            elif isinstance(expr, tuple):
                expr = tuple(self._subexpr(sympify(a), dummies_dict) for a in expr)
            elif isinstance(expr, list):
                expr = [self._subexpr(sympify(a), dummies_dict) for a in expr]
        return expr

    def _print_funcargwrapping(self, args):
        """Generate argument wrapping code.

        args is the argument list of the generated function (strings).

        Return value is a list of lines of code that will be inserted  at
        the beginning of the function definition.
        """
        return []

    def _print_unpacking(self, unpackto, arg):
        """Generate argument unpacking code.

        arg is the function argument to be unpacked (a string), and
        unpackto is a list or nested lists of the variable names (strings) to
        unpack to.
        """
        def unpack_lhs(lvalues):
            return '[{}]'.format(', '.join(
                unpack_lhs(val) if iterable(val) else val for val in lvalues))

        return ['{} = {}'.format(unpack_lhs(unpackto), arg)]

    def _handle_Subs(self, expr, out):
        """Any instance of Subs is extracted and returned as assignment pairs."""
        from sympy.core.basic import Basic
        from sympy.core.function import Subs
        from sympy.core.symbol import Dummy
        from sympy.matrices.matrixbase import MatrixBase

        def _replace(ex, variables, point):
            safe = {}
            for lhs, rhs in zip(variables, point):
                dummy = Dummy()
                safe[lhs] = dummy
                out.append((dummy, rhs))
            return ex.xreplace(safe)

        if isinstance(expr, (Basic, MatrixBase)):
            expr = expr.replace(Subs, _replace)
        elif iterable(expr):
            expr = type(expr)([self._handle_Subs(e, out) for e in expr])
        return expr

class _TensorflowEvaluatorPrinter(_EvaluatorPrinter):
    def _print_unpacking(self, lvalues, rvalue):
        """Generate argument unpacking code.

        This method is used when the input value is not iterable,
        but can be indexed (see issue #14655).
        """

        def flat_indexes(elems):
            for n, el in enumerate(elems):
                if iterable(el):
                    for ndeep in flat_indexes(el):
                        yield (n,) + ndeep
                else:
                    yield (n,)

        indexed = ', '.join('{}[{}]'.format(rvalue, ']['.join(map(str, ind)))
                                for ind in flat_indexes(lvalues))

        return ['[{}] = [{}]'.format(', '.join(flatten(lvalues)), indexed)]

def _imp_namespace(expr, namespace=None):
    """ Return namespace dict with function implementations

    We need to search for functions in anything that can be thrown at
    us - that is - anything that could be passed as ``expr``.  Examples
    include SymPy expressions, as well as tuples, lists and dicts that may
    contain SymPy expressions.

    Parameters
    ----------
    expr : object
       Something passed to lambdify, that will generate valid code from
       ``str(expr)``.
    namespace : None or mapping
       Namespace to fill.  None results in new empty dict

    Returns
    -------
    namespace : dict
       dict with keys of implemented function names within ``expr`` and
       corresponding values being the numerical implementation of
       function

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.utilities.lambdify import implemented_function, _imp_namespace
    >>> from sympy import Function
    >>> f = implemented_function(Function('f'), lambda x: x+1)
    >>> g = implemented_function(Function('g'), lambda x: x*10)
    >>> namespace = _imp_namespace(f(g(x)))
    >>> sorted(namespace.keys())
    ['f', 'g']
    """
    # Delayed import to avoid circular imports
    from sympy.core.function import FunctionClass
    if namespace is None:
        namespace = {}
    # tuples, lists, dicts are valid expressions
    if isinstance(expr, (list, tuple)):
        for arg in expr:
            _imp_namespace(arg, namespace)
        return namespace
    elif isinstance(expr, dict):
        for key, val in expr.items():
            # functions can be in dictionary keys
            _imp_namespace(key, namespace)
            _imp_namespace(val, namespace)
        return namespace
    # SymPy expressions may be Functions themselves
    func = getattr(expr, 'func', None)
    if isinstance(func, FunctionClass):
        imp = getattr(func, '_imp_', None)
        if imp is not None:
            name = expr.func.__name__
            if name in namespace and namespace[name] != imp:
                raise ValueError('We found more than one '
                                 'implementation with name '
                                 '"%s"' % name)
            namespace[name] = imp
    # and / or they may take Functions as arguments
    if hasattr(expr, 'args'):
        for arg in expr.args:
            _imp_namespace(arg, namespace)
    return namespace


def implemented_function(symfunc, implementation):
    """ Add numerical ``implementation`` to function ``symfunc``.

    ``symfunc`` can be an ``UndefinedFunction`` instance, or a name string.
    In the latter case we create an ``UndefinedFunction`` instance with that
    name.

    Be aware that this is a quick workaround, not a general method to create
    special symbolic functions. If you want to create a symbolic function to be
    used by all the machinery of SymPy you should subclass the ``Function``
    class.

    Parameters
    ----------
    symfunc : ``str`` or ``UndefinedFunction`` instance
       If ``str``, then create new ``UndefinedFunction`` with this as
       name.  If ``symfunc`` is an Undefined function, create a new function
       with the same name and the implemented function attached.
    implementation : callable
       numerical implementation to be called by ``evalf()`` or ``lambdify``

    Returns
    -------
    afunc : sympy.FunctionClass instance
       function with attached implementation

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.utilities.lambdify import implemented_function
    >>> from sympy import lambdify
    >>> f = implemented_function('f', lambda x: x+1)
    >>> lam_f = lambdify(x, f(x))
    >>> lam_f(4)
    5
    """
    # Delayed import to avoid circular imports
    from sympy.core.function import UndefinedFunction
    # if name, create function to hold implementation
    kwargs = {}
    if isinstance(symfunc, UndefinedFunction):
        kwargs = symfunc._kwargs
        symfunc = symfunc.__name__
    if isinstance(symfunc, str):
        # Keyword arguments to UndefinedFunction are added as attributes to
        # the created class.
        symfunc = UndefinedFunction(
            symfunc, _imp_=staticmethod(implementation), **kwargs)
    elif not isinstance(symfunc, UndefinedFunction):
        raise ValueError(filldedent('''
            symfunc should be either a string or
            an UndefinedFunction instance.'''))
    return symfunc


def _too_large_for_docstring(expr, limit):
    """Decide whether an ``Expr`` is too large to be fully rendered in a
    ``lambdify`` docstring.

    This is a fast alternative to ``count_ops``, which can become prohibitively
    slow for large expressions, because in this instance we only care whether
    ``limit`` is exceeded rather than counting the exact number of nodes in the
    expression.

    Parameters
    ==========
    expr : ``Expr``, (nested) ``list`` of ``Expr``, or ``Matrix``
        The same objects that can be passed to the ``expr`` argument of
        ``lambdify``.
    limit : ``int`` or ``None``
        The threshold above which an expression contains too many nodes to be
        usefully rendered in the docstring. If ``None`` then there is no limit.

    Returns
    =======
    bool
        ``True`` if the number of nodes in the expression exceeds the limit,
        ``False`` otherwise.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.utilities.lambdify import _too_large_for_docstring
    >>> expr = x
    >>> _too_large_for_docstring(expr, None)
    False
    >>> _too_large_for_docstring(expr, 100)
    False
    >>> _too_large_for_docstring(expr, 1)
    False
    >>> _too_large_for_docstring(expr, 0)
    True
    >>> _too_large_for_docstring(expr, -1)
    True

    Does this split it?

    >>> expr = [x, y, z]
    >>> _too_large_for_docstring(expr, None)
    False
    >>> _too_large_for_docstring(expr, 100)
    False
    >>> _too_large_for_docstring(expr, 1)
    True
    >>> _too_large_for_docstring(expr, 0)
    True
    >>> _too_large_for_docstring(expr, -1)
    True

    >>> expr = [x, [y], z, [[x+y], [x*y*z, [x+y+z]]]]
    >>> _too_large_for_docstring(expr, None)
    False
    >>> _too_large_for_docstring(expr, 100)
    False
    >>> _too_large_for_docstring(expr, 1)
    True
    >>> _too_large_for_docstring(expr, 0)
    True
    >>> _too_large_for_docstring(expr, -1)
    True

    >>> expr = ((x + y + z)**5).expand()
    >>> _too_large_for_docstring(expr, None)
    False
    >>> _too_large_for_docstring(expr, 100)
    True
    >>> _too_large_for_docstring(expr, 1)
    True
    >>> _too_large_for_docstring(expr, 0)
    True
    >>> _too_large_for_docstring(expr, -1)
    True

    >>> from sympy import Matrix
    >>> expr = Matrix([[(x + y + z), ((x + y + z)**2).expand(),
    ...                 ((x + y + z)**3).expand(), ((x + y + z)**4).expand()]])
    >>> _too_large_for_docstring(expr, None)
    False
    >>> _too_large_for_docstring(expr, 1000)
    False
    >>> _too_large_for_docstring(expr, 100)
    True
    >>> _too_large_for_docstring(expr, 1)
    True
    >>> _too_large_for_docstring(expr, 0)
    True
    >>> _too_large_for_docstring(expr, -1)
    True

    """
    # Must be imported here to avoid a circular import error
    from sympy.core.traversal import postorder_traversal

    if limit is None:
        return False
    for i, _ in enumerate(postorder_traversal(expr), 1):
        if i > limit:
            return True
    return False


# Status values reported by lambdify_precheck
PRECHECK_SUPPORTED = "supported"
PRECHECK_UNSUPPORTED = "unsupported"
PRECHECK_UNKNOWN = "unknown"
PRECHECK_ARGUMENT = "argument"

_PRECHECK_STATUS_RANK = {
    PRECHECK_SUPPORTED: 0,
    PRECHECK_ARGUMENT: 0,
    PRECHECK_UNKNOWN: 1,
    PRECHECK_UNSUPPORTED: 2,
}


class LambdifyPrecheckResult:
    """Report returned by :func:`lambdify_precheck`.

    Attributes
    ==========

    backend : str
        Label of the backend the selected printer targets (for instance
        ``"numpy"`` or ``"math"``).
    functions : dict
        Maps each function used by the expression to an entry describing the
        status. An entry has the keys:

        ``status``
            One of ``"supported"``, ``"unsupported"``, ``"unknown"`` and
            ``"argument"`` (the latter for undefined functions that are
            supplied as lambda arguments).
        ``names``
            Maps each bare name that the generated code would look up to a
            ``(status, provider)`` pair.
        ``provider``
        Comma separated list of the namespace sources providing the needed
        names.
        ``top_level``
            Whether the function is used outside of any ``Piecewise``/``ITE``
            branch.
        ``branches``
            Locations (``(branch id, case index)`` pairs) in which the
            function is used, including branches that are not taken on common
            inputs.
    constants : dict
        Like ``functions`` but for symbolic constants (``pi``, ``E``, ...).
    operations : dict
        Additional bare names that the generated code would look up but that
        are not emitted as a function or constant entry (for example
        ``less`` or ``equal`` used to print relational operators). Each value
        carries ``status`` and ``provider`` keys.
    branches : list
        One entry per ``Piecewise`` or ``ITE`` occurrence. Each entry records
        the construct the backend would emit (``construct``), whether all
        cases are evaluated eagerly or lazily (``evaluation``), the status of
        the branch construct, and the requirements appearing in each case.
        Eager backends (such as NumPy) evaluate every case, so a requirement
        missing from a rarely taken case still fails.
    failed_imports : list
        ``(module, name)`` pairs that the printer would import but that the
        backend does not actually provide.
    print_error : tuple or None
        If the whole expression cannot be printed, a ``(status, message)``
        pair describing why.

    """
    def __init__(self, backend, functions, constants, branches, *,
                 operations=(), failed_imports=(), print_error=None):
        self.backend = backend
        self.functions = functions
        self.constants = constants
        self.operations = dict(operations)
        self.branches = branches
        self.failed_imports = list(failed_imports)
        self.print_error = print_error

    @staticmethod
    def _problems(mapping, category):
        problems = []
        for name, entry in mapping.items():
            if entry['status'] in (PRECHECK_UNSUPPORTED, PRECHECK_UNKNOWN):
                problems.append((entry['status'], '%s:%s' % (category, name)))
        return problems

    def _all_problems(self):
        problems = self._problems(self.functions, 'function')
        problems += self._problems(self.constants, 'constant')
        problems += self._problems(self.operations, 'operation')
        for branch in self.branches:
            if branch['status'] in (PRECHECK_UNSUPPORTED, PRECHECK_UNKNOWN):
                problems.append((branch['status'],
                                 'branch:%d' % branch['id']))
        for mod, name in self.failed_imports:
            problems.append((PRECHECK_UNSUPPORTED,
                             'import:%s.%s' % (mod, name)))
        if self.print_error is not None:
            problems.append((self.print_error[0], 'expression'))
        return problems

    @property
    def unsupported(self):
        """Sorted labels of requirements that are provably not mappable."""
        return sorted(label for status, label in self._all_problems()
                      if status == PRECHECK_UNSUPPORTED)

    @property
    def unknown(self):
        """Sorted labels whose mappability cannot be decided statically."""
        return sorted(label for status, label in self._all_problems()
                      if status == PRECHECK_UNKNOWN)

    @property
    def all_supported(self):
        """Whether every requirement was verified as mappable."""
        return not self._all_problems()

    def __repr__(self):
        lines = ['lambdify precheck for the %r backend' % self.backend]
        for title, mapping in (('functions', self.functions),
                               ('constants', self.constants),
                               ('operations', self.operations)):
            if not mapping:
                continue
            lines.append('%s:' % title)
            for name in sorted(mapping):
                entry = mapping[name]
                suffix = (' (%s)' % entry['provider']) if entry.get('provider') else ''
                note = ''
                if entry['status'] in (PRECHECK_UNSUPPORTED, PRECHECK_UNKNOWN) \
                        and entry.get('note'):
                    note = ' - %s' % entry['note']
                lines.append('  %s: %s%s%s' %
                             (name, entry['status'], suffix, note))
        for branch in self.branches:
            lines.append('%s #%d: %s (%s, %s evaluation)' % (
                branch['kind'], branch['id'], branch['status'],
                branch['construct'], branch['evaluation']))
        if self.failed_imports:
            lines.append('failed imports: %s' %
                         ', '.join('%s.%s' % pair for pair in self.failed_imports))
        if self.print_error is not None:
            lines.append('expression: %s - %s' % self.print_error)
        return '\n'.join(lines)


class _PrecheckFunctionPrinter:
    """Adapter making a plain function printer usable by the precheck."""

    def __init__(self, printer):
        self._printer = printer
        self.module_imports = {}

    def doprint(self, expr):
        return self._printer(expr)

    def _print(self, expr):
        return self._printer(expr)


def lambdify_precheck(expr, modules=None, args=None, *, printer=None,
                      use_imps=True):
    """Statically check whether *expr* can be lambdified for a backend.

    Explanation
    ===========

    ``lambdify`` only fails once the generated function is called and a
    missing name is looked up. Expressions containing ``Piecewise`` (or
    ``ITE``) are especially tricky because the cases that are not taken on
    common inputs are still parsed by vectorized backends such as NumPy,
    which evaluate every case. A function that is only used in such a case
    therefore makes the generated function fail on inputs that never reach
    that case.

    This function performs the same namespace and printer setup as
    :func:`lambdify` and then walks the whole expression tree -- including
    every branch case -- without calling the generated function or any of
    the user provided implementations. It reports, one by one, whether the
    functions, constants and branch operations that the expression actually
    needs can be mapped by the selected backend.

    User provided mappings (dictionaries in *modules*, or implementations
    attached through :func:`implemented_function`) take part with the same
    priority as in :func:`lambdify`: an opaque user provided object makes the
    requirement ``"unknown"``, because its compatibility with the backend
    cannot be verified statically; it is never called.

    Parameters
    ==========

    expr : Expr
        The expression (or a nested list/tuple/dict of expressions) that
        would be passed to :func:`lambdify`.
    modules :
        The same ``modules`` argument as accepted by :func:`lambdify`.
    args : optional
        The lambda arguments, as accepted by :func:`lambdify`. Undefined
        functions appearing here are reported as ``"argument"`` instead of
        being looked up in the backend namespace. If omitted, undefined
        functions that cannot be resolved are reported as ``"unknown"``.
    printer : optional
        A custom printer, as accepted by :func:`lambdify`.
    use_imps : bool, optional
        Whether implementations attached through ``_imp_`` are taken into
        account (highest priority), like in :func:`lambdify`.

    Returns
    =======

    LambdifyPrecheckResult
        Structured report. See :class:`LambdifyPrecheckResult`. The
        convenience attributes ``all_supported``, ``unsupported`` and
        ``unknown`` summarize the result.

    Examples
    ========

    >>> from sympy import symbols, sin, besselj, Piecewise
    >>> from sympy.utilities.lambdify import lambdify_precheck
    >>> x = symbols('x')

    The second case is inspected even though common inputs do not reach it:

    >>> result = lambdify_precheck(
    ...     Piecewise((sin(x), x < 1), (besselj(0, x), True)), modules='math')
    >>> result.functions['besselj']['status']
    'unsupported'
    >>> result.branches[0]['evaluation']
    'lazy'
    >>> result.all_supported
    False

    A user provided mapping has priority over the backend mapping, but an
    opaque implementation cannot be verified and is reported as unknown:

    >>> result = lambdify_precheck(
    ...     besselj(0, x), modules=[{'besselj': lambda *args: args}, 'math'])
    >>> result.functions['besselj']['status']
    'unknown'

    Plain backends otherwise report everything as supported:

    >>> from sympy import E, I, pi
    >>> lambdify_precheck(pi + E + I, modules='math').all_supported
    True

    """
    from sympy.core.basic import Basic
    from sympy.core.function import (AppliedUndef, Application, Derivative,
                                    Function)
    from sympy.core.numbers import (ComplexInfinity, ImaginaryUnit, Infinity,
                                    NaN, NegativeInfinity, NumberSymbol)
    from sympy.core.symbol import Symbol
    from sympy.concrete.summations import Sum
    from sympy.functions.elementary.piecewise import Piecewise
    from sympy.integrals.integrals import Integral
    from sympy.logic.boolalg import ITE
    from sympy.printing.codeprinter import (CodePrinter,
                                            PrintMethodNotImplementedError)
    from sympy.printing.lambdarepr import NumExprPrinter
    from sympy.printing.numpy import NumPyPrinter, SciPyPrinter
    from sympy.printing.pycode import (AbstractPythonCodePrinter, CmathPrinter,
                                       MpmathPrinter, SymPyPrinter)
    from sympy.printing.pytorch import TorchPrinter
    from sympy.printing.tensorflow import TensorflowPrinter

    if modules is None:
        modules = _default_modules()
    specs = _collect_namespaces(expr, modules, use_imps=False)
    imp_namespace = _imp_namespace(expr) if use_imps else {}
    namespaces = ([imp_namespace] if use_imps else []) + specs

    # Symbols occurring in the expression are lambda inputs, not backend
    # requirements.
    bound_symbols = set()
    if hasattr(expr, "atoms"):
        for term in expr.atoms(Symbol):
            bound_symbols.add(str(term))

    # Matrix symbols (and similar named leaves) can also be passed as lambda
    # arguments; without ``args`` their status cannot be decided statically.
    from sympy.matrices.expressions import MatrixExpr
    possible_inputs = set()
    if hasattr(expr, "atoms"):
        for term in expr.atoms(MatrixExpr):
            name = str(term)
            if isinstance(term, Basic) and name.isidentifier() \
                    and not keyword.iskeyword(name):
                possible_inputs.add(name)

    bound_functions = set()
    if args is not None:
        _collect_bound_arguments(args, bound_symbols, bound_functions)

    # Namespace sources in priority order, with the provenance of each name.
    sources = []
    if use_imps:
        sources.append(("implemented function (_imp_)", "imp", imp_namespace))
    for index, spec in enumerate(specs):
        ns = _get_namespace(spec)
        if isinstance(spec, dict):
            label = "user-provided mapping #%d" % index
            kind = "user"
        elif isinstance(spec, str):
            label, kind = spec, "backend"
        else:
            label, kind = getattr(spec, "__name__", "module"), "backend"
        sources.append((label, kind, ns))

    providers = {}
    backend_values = {}
    for label, kind, ns in sources:
        for key, value in ns.items():
            providers.setdefault(key, (label, kind, value))
            if kind == "backend":
                backend_values.setdefault(id(value), label)
    # lambdify always adds these to the generated function's namespace.
    providers.setdefault('builtins', ("builtins module", "backend", builtins))
    providers.setdefault('range', ("Python builtin", "backend", range))

    # Build the printer exactly like lambdify would.
    if printer is None:
        Printer = _select_printer_class(namespaces)
        user_functions = _user_functions_from_namespaces(namespaces)
        printer = Printer({'fully_qualified_modules': False, 'inline': True,
                           'allow_unknown_functions': True,
                           'user_functions': user_functions})
    elif inspect.isclass(printer):
        printer = printer()
    elif inspect.isfunction(printer):
        printer = _PrecheckFunctionPrinter(printer)

    is_numexpr = isinstance(printer, NumExprPrinter)

    def iter_nodes(node):
        if isinstance(node, Basic):
            yield node
            for subnode in node.args:
                yield from iter_nodes(subnode)
        elif isinstance(node, (list, tuple)):
            for subnode in node:
                yield from iter_nodes(subnode)
        elif isinstance(node, dict):
            for key, value in node.items():
                yield from iter_nodes(key)
                yield from iter_nodes(value)

    # NumExprPrinter inlines _imp_ implementations while printing, i.e. it
    # would call the user provided implementation. Such functions are reported
    # as unknown and their printing is skipped entirely.
    numexpr_table = getattr(printer, '_numexpr_functions', {}) \
        if is_numexpr else {}
    inlined_user_imps = set()
    if is_numexpr:
        for node in iter_nodes(expr):
            if isinstance(node, Function) \
                    and type(node).__name__ not in numexpr_table \
                    and getattr(getattr(node, 'func', None),
                                '_imp_', None) is not None:
                inlined_user_imps.add(type(node).__name__)

    # Build the exact function string lambdify would generate (printing only
    # produces strings; user provided numerical implementations are never
    # called). This registers the module imports that lambdify would
    # execute, and the signature names are known bound names.
    print_error = None
    printed = None
    signature_names = set()
    if inlined_user_imps:
        print_error = (
            PRECHECK_UNKNOWN,
            "numexpr inlines the user provided implementation(s) of %s, "
            "which cannot be verified statically"
            % ', '.join(sorted(inlined_user_imps)))
    else:
        if args is None:
            iterable_args = list(expr.atoms(Symbol)) \
                if hasattr(expr, 'atoms') else []
        elif isinstance(args, Basic):
            iterable_args = (args,)
        else:
            iterable_args = args
        try:
            funcprinter = _EvaluatorPrinter(printer, dummify=False)
            printed = funcprinter.doprint(
                '_lambdifygenerated', iterable_args, expr)
            signature = printed.split(':', 1)[0]
            signature_names = _extract_identifiers(signature)
            signature_names.discard('def')
        except PrintMethodNotImplementedError as exc:
            print_error = (PRECHECK_UNSUPPORTED, str(exc))
        except NotImplementedError as exc:
            print_error = (PRECHECK_UNSUPPORTED, str(exc))
        except TypeError as exc:
            severity = (PRECHECK_UNSUPPORTED if is_numexpr
                        else PRECHECK_UNKNOWN)
            print_error = (severity, str(exc))
        except Exception as exc:  # noqa: BLE001
            print_error = (PRECHECK_UNKNOWN,
                           '%s: %s' % (type(exc).__name__, exc))
    bound_symbols |= signature_names

    failed_imports = []
    for mod, keys in (getattr(printer, 'module_imports', None) or {}).items():
        try:
            imported = importlib.import_module(mod)
        except ImportError:
            failed_imports.extend((mod, key) for key in keys)
            continue
        for key in keys:
            if hasattr(imported, key):
                providers.setdefault(
                    key, ('%s (import)' % mod, "backend",
                          getattr(imported, key)))
            else:
                failed_imports.append((mod, key))

    def backend_label():
        if is_numexpr:
            return "numexpr"
        if isinstance(printer, SciPyPrinter):
            return "scipy"
        if isinstance(printer, NumPyPrinter):
            return getattr(printer, '_module', "numpy")
        if isinstance(printer, TensorflowPrinter):
            return "tensorflow"
        if isinstance(printer, TorchPrinter):
            return "torch"
        if isinstance(printer, MpmathPrinter):
            return "mpmath"
        if isinstance(printer, SymPyPrinter):
            return "sympy"
        if isinstance(printer, CmathPrinter):
            return "cmath"
        if isinstance(printer, AbstractPythonCodePrinter):
            for spec in specs:
                if isinstance(spec, str):
                    return spec
                name = getattr(spec, '__name__', None)
                if name:
                    return name
            return "python"
        return type(printer).__name__

    def branch_spec():
        if is_numexpr:
            return ("numexpr.where", "eager", PRECHECK_SUPPORTED)
        if isinstance(printer, NumPyPrinter):
            return ('%s.select' % getattr(printer, '_module', "numpy"),
                    "eager", PRECHECK_SUPPORTED)
        if isinstance(printer, TensorflowPrinter):
            return ("tensorflow.where", "eager", PRECHECK_SUPPORTED)
        if isinstance(printer, TorchPrinter):
            return ("torch.where", "eager", PRECHECK_SUPPORTED)
        if isinstance(printer, AbstractPythonCodePrinter):
            return ("conditional expression (if/else)", "lazy",
                    PRECHECK_SUPPORTED)
        method = getattr(printer, '_print_Piecewise', None)
        if method is None or \
                getattr(method, '__func__', method) is CodePrinter._print_not_supported:
            return (None, None, PRECHECK_UNSUPPORTED)
        return ("custom piecewise printer", "unknown", PRECHECK_UNKNOWN)

    def resolve_name(name, allow_compiler=False):
        if name in bound_symbols:
            return PRECHECK_ARGUMENT, "lambda argument"
        provider = providers.get(name)
        if provider is not None:
            label, kind, value = provider
            if kind == "backend":
                return PRECHECK_SUPPORTED, label
            if id(value) in backend_values:
                return PRECHECK_SUPPORTED, '%s -> %s' % (
                    label, backend_values[id(value)])
            return PRECHECK_UNKNOWN, label
        if hasattr(builtins, name):
            return PRECHECK_SUPPORTED, "Python builtin"
        if allow_compiler:
            return PRECHECK_UNKNOWN, "numexpr expression compiler"
        return PRECHECK_UNSUPPORTED, None

    functions = {}
    constants = {}
    branches = []
    function_like = (Function, Application, Derivative, Integral, Sum)
    call_name_cache = {}

    def print_node(node):
        try:
            text = printer._print(node)
        except PrintMethodNotImplementedError as exc:
            return None, (PRECHECK_UNSUPPORTED, str(exc))
        except NotImplementedError as exc:
            return None, (PRECHECK_UNSUPPORTED, str(exc))
        except TypeError as exc:
            severity = PRECHECK_UNSUPPORTED if is_numexpr else PRECHECK_UNKNOWN
            return None, (severity, str(exc))
        except Exception as exc:  # noqa: BLE001
            return None, (PRECHECK_UNKNOWN,
                          '%s: %s' % (type(exc).__name__, exc))
        return text, None

    def child_nodes(node):
        if isinstance(node, Basic):
            return list(node.args)
        if isinstance(node, (list, tuple)):
            return list(node)
        if isinstance(node, dict):
            return list(node.keys()) + list(node.values())
        return []

    def own_call_names(node):
        """Call-position names emitted by *node* itself.

        Calls emitted by function-like subexpressions of its arguments are
        excluded: they are reported on their own entries.
        """
        if node in call_name_cache:
            return call_name_cache[node]
        if is_numexpr:
            # Printing this node would print (and therefore inline-call) a
            # descendant carrying a user provided _imp_ implementation.
            for subnode in iter_nodes(node):
                if isinstance(subnode, Function) \
                        and type(subnode).__name__ not in numexpr_table \
                        and getattr(getattr(subnode, 'func', None),
                                    '_imp_', None) is not None:
                    problem = (PRECHECK_UNKNOWN,
                               "implemented_function inlined by numexpr")
                    call_name_cache[node] = (set(), problem)
                    return set(), problem
        text, problem = print_node(node)
        if problem is not None:
            call_name_cache[node] = (set(), problem)
            return set(), problem
        nested = set()
        stack = child_nodes(node)
        seen = set()
        while stack:
            child = stack.pop()
            if not isinstance(child, Basic) or id(child) in seen:
                continue
            seen.add(id(child))
            if isinstance(child, function_like):
                nested |= own_call_names(child)[0]
            stack.extend(child_nodes(child))
        names = _extract_identifiers(
            text, calls_only=True,
            bound=bound_symbols | bound_functions) - nested
        call_name_cache[node] = (names, None)
        return names, None

    def record(bucket, key, ctx, *, status=None, names=None, note=None):
        entry = bucket.get(key)
        if entry is None:
            entry = {
                'status': PRECHECK_SUPPORTED,
                'names': {},
                'provider': '',
                'note': None,
                'top_level': False,
                'branches': [],
                '_statuses': set(),
            }
            bucket[key] = entry
        if ctx:
            if ctx not in entry['branches']:
                entry['branches'].append(ctx)
        else:
            entry['top_level'] = True
        if status is not None:
            entry['_statuses'].add(status)
        if note is not None and entry['note'] is None:
            entry['note'] = note
        for name, result in (names or {}).items():
            previous = entry['names'].get(name)
            if previous is None or \
                    _PRECHECK_STATUS_RANK[result[0]] > _PRECHECK_STATUS_RANK[previous[0]]:
                entry['names'][name] = result
                entry['_statuses'].add(result[0])

    def classify_names(names, allow_compiler=False):
        results = {}
        for name in sorted(names):
            if name in bound_symbols:
                continue
            results[name] = resolve_name(name, allow_compiler)
        return results

    def handle_function(node, ctx):
        key = type(node).__name__
        if isinstance(node, AppliedUndef) and key in bound_functions:
            record(functions, key, ctx, status=PRECHECK_ARGUMENT,
                   note="supplied as a lambda argument")
            return
        if is_numexpr:
            if key in numexpr_table:
                emitted = numexpr_table[key]
                record(functions, key, ctx, status=PRECHECK_SUPPORTED,
                       names={emitted: (PRECHECK_SUPPORTED,
                                        "numexpr built-in function")})
                return
            if getattr(getattr(node, 'func', None), '_imp_', None) is not None:
                # numexpr inlines this implementation; printing it would call
                # the user provided function.
                record(functions, key, ctx, status=PRECHECK_UNKNOWN,
                       note="implemented_function inlined by numexpr")
                return
        names, print_problem = own_call_names(node)
        if print_problem is not None:
            record(functions, key, ctx, status=print_problem[0],
                   note=print_problem[1])
            return
        results = classify_names(names)
        status = None
        note = None
        if isinstance(node, AppliedUndef):
            own = results.pop(key, None)
            if own is not None and own[0] != PRECHECK_UNSUPPORTED:
                # The undefined function name resolves through a mapping.
                results[key] = own
            else:
                # Not provided anywhere; it might still be supplied as a
                # lambda argument, which cannot be known without ``args``.
                status = (PRECHECK_UNKNOWN if args is None
                          else PRECHECK_UNSUPPORTED)
                note = ("undefined function; it must be supplied as a lambda "
                        "argument or through a module mapping")
        record(functions, key, ctx, status=status, names=results, note=note)

    def handle_constant(node, ctx):
        key = type(node).__name__
        text, print_problem = print_node(node)
        if print_problem is not None:
            record(constants, key, ctx, status=print_problem[0],
                   note=print_problem[1])
            return
        # A constant compiled into a literal (1j, float('inf'), inlined
        # NumberSymbol) needs no namespace name and is always supported.
        results = classify_names(
            _extract_identifiers(text), allow_compiler=is_numexpr)
        record(constants, key, ctx, names=results)

    def walk(node, ctx=()):
        if isinstance(node, Piecewise):
            branch_id = len(branches)
            construct, evaluation, status = branch_spec()
            branches.append({
                'id': branch_id,
                'kind': 'Piecewise',
                'construct': construct,
                'evaluation': evaluation,
                'status': status,
                'cases': [{'index': i, 'condition': str(pair.cond)}
                          for i, pair in enumerate(node.args)],
            })
            for i, pair in enumerate(node.args):
                child_ctx = ctx + (branch_id, i)
                walk(pair.expr, child_ctx)
                walk(pair.cond, child_ctx)
            return
        if isinstance(node, ITE):
            branch_id = len(branches)
            construct, evaluation, status = branch_spec()
            branches.append({
                'id': branch_id,
                'kind': 'ITE',
                'construct': construct,
                'evaluation': evaluation,
                'status': status,
                'cases': [
                    {'index': 1, 'condition': str(node.args[0])},
                    {'index': 2, 'condition': 'else'},
                ],
            })
            walk(node.args[0], ctx + (branch_id, 1))
            walk(node.args[1], ctx + (branch_id, 1))
            walk(node.args[2], ctx + (branch_id, 2))
            return
        if isinstance(node, Basic):
            if isinstance(node, function_like):
                handle_function(node, ctx)
            elif isinstance(node, (NumberSymbol, ImaginaryUnit, NaN,
                                   Infinity, NegativeInfinity,
                                   ComplexInfinity)):
                handle_constant(node, ctx)
            for subnode in node.args:
                walk(subnode, ctx)
        elif isinstance(node, (list, tuple)):
            for subnode in node:
                walk(subnode, ctx)
        elif isinstance(node, dict):
            for key, value in node.items():
                walk(key, ctx)
                walk(value, ctx)

    walk(expr)

    for bucket in (functions, constants):
        for entry in bucket.values():
            statuses = entry['_statuses']
            if statuses:
                entry['status'] = max(
                    statuses, key=lambda s: _PRECHECK_STATUS_RANK[s])
            entry['provider'] = ', '.join(sorted({
                provider for result in entry['names'].values()
                for provider in [result[1]] if provider}))
            del entry['_statuses']

    for branch in branches:
        branch_id = branch['id']
        requirements = []
        for category, bucket in (('function', functions),
                                 ('constant', constants)):
            for name, entry in bucket.items():
                cases = sorted({
                    path[j + 1]
                    for path in entry['branches']
                    for j in range(0, len(path) - 1, 2)
                    if path[j] == branch_id
                })
                if cases:
                    requirements.append({
                        'category': category,
                        'name': name,
                        'status': entry['status'],
                        'cases': cases,
                    })
        for case in branch['cases']:
            case_index = case['index']
            case['requirements'] = [
                '%s:%s' % (item['category'], item['name'])
                for item in requirements if case_index in item['cases']
            ]
            case['unsupported'] = sorted(
                item['name'] for item in requirements
                if case_index in item['cases']
                and item['status'] == PRECHECK_UNSUPPORTED)
            case['unknown'] = sorted(
                item['name'] for item in requirements
                if case_index in item['cases']
                and item['status'] == PRECHECK_UNKNOWN)
        branch['unsupported'] = sorted({
            item['name'] for item in requirements
            if item['status'] == PRECHECK_UNSUPPORTED})
        branch['unknown'] = sorted({
            item['name'] for item in requirements
            if item['status'] == PRECHECK_UNKNOWN})

    # Safety net over the full generated function string: any bare name that
    # would be looked up at runtime but was not attributed above (for example
    # ``less``/``equal`` used by relational printing) is checked here.
    operations = {}
    if printed is not None:
        attributed = set()
        for bucket in (functions, constants):
            for entry in bucket.values():
                attributed.update(entry['names'])
        unattributed = _extract_identifiers(printed, include_heads=True) \
            - bound_symbols - bound_functions - attributed
        for name in sorted(unattributed):
            if name in possible_inputs:
                status = (PRECHECK_UNKNOWN if args is None
                          else PRECHECK_UNSUPPORTED)
                provider = 'possible lambda argument' if args is None else ''
            else:
                status, provider = resolve_name(
                    name, allow_compiler=is_numexpr)
            operations[name] = {'status': status, 'provider': provider or ''}

    return LambdifyPrecheckResult(
        backend_label(), functions, constants, branches,
        operations=operations, failed_imports=failed_imports,
        print_error=print_error)


def _collect_bound_arguments(args, bound_symbols, bound_functions):
    """Collect symbol and undefined function names bound as lambda arguments."""
    from sympy.core.basic import Basic
    from sympy.core.function import (AppliedUndef, Derivative, Function,
                                     UndefinedFunction)
    from sympy.core.symbol import Symbol
    from sympy.matrices import DeferredVector
    from sympy.matrices.expressions import MatrixExpr

    def rec(arg):
        if inspect.isclass(arg) and isinstance(arg, UndefinedFunction):
            bound_functions.add(arg.__name__)
        elif isinstance(arg, DeferredVector):
            bound_symbols.add(str(arg))
        elif isinstance(arg, Basic):
            if isinstance(arg, AppliedUndef):
                bound_functions.add(arg.func.__name__)
            if isinstance(arg, (Function, Derivative)):
                bound_symbols.update(map(str, arg.atoms(Symbol)))
            elif isinstance(arg, (Symbol, MatrixExpr)):
                bound_symbols.add(str(arg))
            for subarg in arg.args:
                rec(subarg)
        elif isinstance(arg, str):
            bound_symbols.add(arg)
        elif iterable(arg, exclude=(str, bytes)):
            for subarg in arg:
                rec(subarg)

    if isinstance(args, (Basic, str)) or inspect.isclass(args) or \
            not iterable(args, exclude=(str, bytes)):
        rec(args)
    else:
        for arg in args:
            rec(arg)


def _extract_identifiers(code, calls_only=False, bound=(), include_heads=False):
    """Return the bare identifiers referenced by a snippet of generated code.

    Attribute names on either side of a dot, keywords, keyword argument names
    and comprehension variables are excluded. When *calls_only* is true, only
    identifiers used in call position (immediately followed by ``(``) are
    returned, and names in *bound* are skipped. When *include_heads* is true,
    the leading name of a dotted attribute access is also returned.
    """
    bound = set(bound)
    ignored = {
        tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
        tokenize.ENCODING, tokenize.COMMENT, tokenize.ENDMARKER,
    }
    try:
        tokens = [token for token in
                  tokenize.generate_tokens(io.StringIO(code).readline)
                  if token.type not in ignored]
    except tokenize.TokenError:
        return set()

    local_names = set()
    i = 0
    while i < len(tokens):
        if tokens[i].type == tokenize.NAME and tokens[i].string == 'for':
            j = i + 1
            while j < len(tokens) and tokens[j].type != tokenize.NAME:
                j += 1
            while j < len(tokens) and tokens[j].string != 'in':
                if tokens[j].type == tokenize.NAME:
                    local_names.add(tokens[j].string)
                j += 1
        i += 1

    names = set()
    for i, token in enumerate(tokens):
        if token.type != tokenize.NAME:
            continue
        name = token.string
        if keyword.iskeyword(name) or name in local_names or name in bound:
            continue
        previous = tokens[i - 1].string if i > 0 else ''
        following = tokens[i + 1].string if i + 1 < len(tokens) else ''
        if previous == '.':
            # Attribute name.
            continue
        if include_heads and following == '.':
            # Leading name of a dotted attribute access.
            names.add(name)
            continue
        if following == '.':
            continue
        if calls_only:
            if following != '(':
                continue
        elif following == '=':
            # Keyword argument name.
            continue
        names.add(name)
    return names
