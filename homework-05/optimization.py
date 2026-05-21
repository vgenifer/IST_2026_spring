import numpy as np
from numpy.linalg import LinAlgError
import scipy
from datetime import datetime
from collections import defaultdict


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        if self._method == 'Constant':
            return self.c

        elif self._method == 'Armijo':
            alpha = previous_alpha if previous_alpha is not None else self.alpha_0
            phi_0 = oracle.func_directional(x_k, d_k, 0)
            dphi_0 = oracle.grad_directional(x_k, d_k, 0)

            # Backtracking: halve alpha until Armijo condition is satisfied
            while oracle.func_directional(x_k, d_k, alpha) > phi_0 + self.c1 * alpha * dphi_0:
                alpha /= 2.0
                if alpha < 1e-15:
                    return None

            return alpha

        elif self._method == 'Wolfe':
            # Try scipy's line search for strong Wolfe conditions
            phi = lambda a: oracle.func_directional(x_k, d_k, a)
            dphi = lambda a: oracle.grad_directional(x_k, d_k, a)

            alpha, *_ = scipy.optimize.line_search(
                phi, dphi,
                alpha0=self.alpha_0,
                c1=self.c1,
                c2=self.c2
            )

            # Fall back to Armijo if Wolfe search failed
            if alpha is None:
                alpha = self.alpha_0 if previous_alpha is None else previous_alpha
                phi_0 = oracle.func_directional(x_k, d_k, 0)
                dphi_0 = oracle.grad_directional(x_k, d_k, 0)

                while oracle.func_directional(x_k, d_k, alpha) > phi_0 + self.c1 * alpha * dphi_0:
                    alpha /= 2.0
                    if alpha < 1e-15:
                        return None

            return alpha

        return None


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradient descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    start_time = datetime.now()
    grad_0 = oracle.grad(x_0)
    grad_0_norm_sq = np.dot(grad_0, grad_0)

    alpha = None  # track previous alpha for warm-starting

    for iteration in range(max_iter + 1):
        f_k = oracle.func(x_k)
        g_k = oracle.grad(x_k)

        # Check for computational errors
        if not np.isfinite(f_k) or not np.all(np.isfinite(g_k)):
            return x_k, 'computational_error', history

        grad_norm = np.linalg.norm(g_k)

        # Record history
        if trace:
            elapsed = (datetime.now() - start_time).total_seconds()
            history['time'].append(elapsed)
            history['func'].append(f_k)
            history['grad_norm'].append(grad_norm)
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))

        if display:
            print(f'Iter {iteration}: f={f_k:.6f}, ||grad||={grad_norm:.6f}')

        # Stopping criterion: ||g(x_k)||^2 <= tolerance * ||g(x_0)||^2
        if grad_norm ** 2 <= tolerance * grad_0_norm_sq:
            return x_k, 'success', history

        if iteration == max_iter:
            break

        # Descent direction: steepest descent
        d_k = -g_k

        # Line search
        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=alpha)

        if alpha is None:
            return x_k, 'computational_error', history

        x_k = x_k + alpha * d_k

        if not np.all(np.isfinite(x_k)):
            return x_k, 'computational_error', history

    return x_k, 'iterations_exceeded', history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    start_time = datetime.now()
    grad_0 = oracle.grad(x_0)
    grad_0_norm_sq = np.dot(grad_0, grad_0)

    for iteration in range(max_iter + 1):
        f_k = oracle.func(x_k)
        g_k = oracle.grad(x_k)
        H_k = oracle.hess(x_k)

        # Check for computational errors
        if not np.isfinite(f_k) or not np.all(np.isfinite(g_k)) or not np.all(np.isfinite(H_k)):
            return x_k, 'computational_error', history

        grad_norm = np.linalg.norm(g_k)

        # Record history
        if trace:
            elapsed = (datetime.now() - start_time).total_seconds()
            history['time'].append(elapsed)
            history['func'].append(f_k)
            history['grad_norm'].append(grad_norm)
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))

        if display:
            print(f'Iter {iteration}: f={f_k:.6f}, ||grad||={grad_norm:.6f}')

        # Stopping criterion
        if grad_norm ** 2 <= tolerance * grad_0_norm_sq:
            return x_k, 'success', history

        if iteration == max_iter:
            break

        # Solve H_k * d_k = -g_k for the Newton direction
        try:
            d_k = np.linalg.solve(H_k, -g_k)
        except LinAlgError:
            return x_k, 'newton_direction_error', history

        # Verify the direction is a descent direction (H_k must be positive definite)
        if np.dot(g_k, d_k) >= 0:
            return x_k, 'newton_direction_error', history

        # Line search
        alpha = line_search_tool.line_search(oracle, x_k, d_k)

        if alpha is None:
            return x_k, 'computational_error', history

        x_k = x_k + alpha * d_k

        if not np.all(np.isfinite(x_k)):
            return x_k, 'computational_error', history

    return x_k, 'iterations_exceeded', history
