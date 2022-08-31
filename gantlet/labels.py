import typing as t

import numpy as np
from numpy import typing as npt



class _Labels:
    def __init__(
        self, 
        factory: t.Callable[[], npt.NDArray],
    ) -> None:
        pass
        self._factory = factory
        self._data = factory()

    def __repr__(self) -> str:
        return self._data.__repr__()
    
    def __str__(self) -> str:
        return self._data.__str__()
    
    def smooth(self, mean: float = 0.0, std: float = 0.01, bounds: t.Tuple[int, int] = None) -> "_Labels":
        """Applies smoothing to labels by adding noise sampled from a random normal distribution.
        The smoothing operation is performed inplace. 

        Args:
            mean (float, optional): mean of distribution. Defaults to 0.0.
            std (float, optional): standard deviation of distribution. Defaults to 0.01.
            bounds (t.Tuple[int, int], optional): min and max bounds to clip values to. Defaults to None.
        """
        noise_shape = self._data.shape
        noise = np.random.normal(mean, std, noise_shape)
        self._data = self._data + noise
        if bounds is not None:
            self._data = np.clip(self._data, bounds[0], bounds[1])
        
        return self

    def flip(self, prob: float = 0.03) -> "_Labels":
        """Replaces a random value `v` with `abs(1 - v)`. Indices of values to flip are 
        sampled from a uniform U(0, 1) distribution. Operation is performed inplace.

        Args:
            prob (float, optional): probability of flipping the value at a given index. Defaults to 0.03.
        """
        samples = np.random.uniform(0., 1., self._data.shape)
        self._data = np.where(samples > prob, self._data, np.abs(1 - self._data))
        return self
    
    def reset(self) -> "_Labels":
        """Reset values to their original view. Operation is performed inplace.
        """
        self._data = self._factory()
        return self
    

class Ones(_Labels):
    def __init__(
        self, 
        shape: t.Tuple[int, ...],
        dtype: str = "float32"
    ) -> None:
        """A class for `ones` or real labels.

        Args:
            shape (t.Tuple[int, ...]): shape of label array.
            dtype (str, optional): dtype of values in array. Defaults to "float32".
        """
        factory = lambda: np.ones(shape, dtype=dtype) 
        super().__init__(factory)


class Zeros(_Labels):
    def __init__(
        self, 
        shape: t.Tuple[int, ...],
        dtype: str = "float32"
    ) -> None:
        """A class for `zeros` or fake labels.

        Args:
            shape (t.Tuple[int, ...]): shape of label array.
            dtype (str, optional): dtype of values in array. Defaults to "float32".
        """
        factory = lambda: np.zeros(shape, dtype=dtype) 
        super().__init__(factory)
    
