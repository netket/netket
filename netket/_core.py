# Copyright 2019 The Simons Foundation, Inc. - All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
import functools
import inspect


def deprecated(reason=None):
    """
    This is a decorator which can be used to mark functions as deprecated. It
    will result in a warning being emitted when the function is used.
    """

    def decorator(func):
        object_type = "class" if inspect.isclass(func) else "function"
        message = "Call to deprecated {} {!r}".format(object_type, func.__name__)
        if reason is not None:
            message += " ({})".format(reason)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(message, category=FutureWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator
