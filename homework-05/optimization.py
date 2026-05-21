import numpy as np
from numpy.linalg import LinAlgError
import scipy
from datetime import datetime
from collections import defaultdict


class LineSearchTool(object):
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
        if self._method == 'Constant':
            return self.c

        elif self._method == 'Armijo':
            alpha = previous_alpha if previous_alpha is not None else self.alpha_0
            phi_0 = oracle.func_directional(x_k, d_k, 0)
            dphi_0 = oracle.grad_directional(x_k, d_k, 0)
            while oracle.func_directional(x_k, d_k, alpha) > phi_0 + self.c1 * alpha * dphi_0:
                alpha /= 2.0
                if alpha < 1e-15:
                    return None
            return alpha

        elif self._method == 'Wolfe':
            phi = lambda a: oracle.func_directional(x_k, d_k, a)
            dphi = lambda a: oracle.grad_directional(x_k, d_k, a)
            alpha, *_ = scipy.optimize._linesearch.line_search_wolfe2(oracle.func, oracle.grad, x_k, d_k, c1=self.c1, c2=self.c2)
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
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    start_time = datetime.now()
    grad_0 = oracle.grad(x_0)
    grad_0_norm_sq = np.dot(grad_0, grad_0)
    alpha = None

    for iteration in range(max_iter + 1):
        f_k = oracle.func(x_k)
        g_k = oracle.grad(x_k)
        if not np.isfinite(f_k) or not np.all(np.isfinite(g_k)):
            return x_k, 'computational_error', history
        grad_norm = np.linalg.norm(g_k)
        if trace:
            elapsed = (datetime.now() - start_time).total_seconds()
            history['time'].append(elapsed)
            history['func'].append(f_k)
            history['grad_norm'].append(grad_norm)
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))
        if display:
            print(f'Iter {iteration}: f={f_k:.6f}, ||grad||={grad_norm:.6f}')
        if grad_norm ** 2 <= tolerance * grad_0_norm_sq:
            return x_k, 'success', history
        if iteration == max_iter:
            break
        d_k = -g_k
        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=alpha)
        if alpha is None:
            return x_k, 'computational_error', history
        x_k = x_k + alpha * d_k
        if not np.all(np.isfinite(x_k)):
            return x_k, 'computational_error', history

    return x_k, 'iterations_exceeded', history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
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
        if not np.isfinite(f_k) or not np.all(np.isfinite(g_k)) or not np.all(np.isfinite(H_k)):
            return x_k, 'computational_error', history
        grad_norm = np.linalg.norm(g_k)
        if trace:
            elapsed = (datetime.now() - start_time).total_seconds()
            history['time'].append(elapsed)
            history['func'].append(f_k)
            history['grad_norm'].append(grad_norm)
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))
        if display:
            print(f'Iter {iteration}: f={f_k:.6f}, ||grad||={grad_norm:.6f}')
        if grad_norm ** 2 <= tolerance * grad_0_norm_sq:
            return x_k, 'success', history
        if iteration == max_iter:
            break
        try:
            d_k = np.linalg.solve(H_k, -g_k)
        except LinAlgError:
            return x_k, 'newton_direction_error', history
        if np.dot(g_k, d_k) >= 0:
            return x_k, 'newton_direction_error', history
        alpha = line_search_tool.line_search(oracle, x_k, d_k)
        if alpha is None:
            return x_k, 'computational_error', history
        x_k = x_k + alpha * d_k
        if not np.all(np.isfinite(x_k)):
            return x_k, 'computational_error', history

    return x_k, 'iterations_exceeded', history
