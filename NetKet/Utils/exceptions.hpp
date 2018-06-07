// Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NETKET_EXCEPTIONS_HPP
#define NETKET_EXCEPTIONS_HPP

#include <exception>
#include <string>

namespace netket {

class NetketBaseException : public std::exception
{
    std::string message_;

public:
    explicit NetketBaseException(const std::string& message)
        : message_(message)
    {
    }

    const char* what() const noexcept override
    {
        return message_.c_str();
    }
};

class InvalidInputError : public NetketBaseException
{
public:
    explicit InvalidInputError(const std::string& message)
        : NetketBaseException(message)
    {
    }
};

}

#endif // NETKET_EXCEPTIONS_HPP
