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

#include <catch.hpp>

#include "netket.hpp"

using namespace netket;

/**
 * Matcher class for checking an exeption message.
 */
template<class Exception>
class HasMessage : public Catch::MatcherBase<Exception>
{
    std::string message_;

public:
    explicit HasMessage(const std::string& message)
        : message_(message)
    {}

    bool match(const Exception& e) const override {
        return message_ == e.what();
    }

    std::string describe() const override {
        std::ostringstream s;
        s << "has error message \"" << message_ << "\"";
        return s.str();
    }
};

TEST_CASE("JSON helper functions throw errors", "[stats]")
{
    SECTION("CheckFieldExists")
    {
        auto pars = json();

        REQUIRE_THROWS_AS(CheckFieldExists(pars, "Key"), InvalidInputError);
        REQUIRE_THROWS_MATCHES(
                    CheckFieldExists(pars, "Key"),
                    InvalidInputError,
                    HasMessage<InvalidInputError>("Field 'Key' is not defined in the input"));

        pars["Key"] = {{"SubKey1", 0}, {"SubKey2", 1}};

        REQUIRE_NOTHROW(CheckFieldExists(pars["Key"], "SubKey1"));
        REQUIRE_NOTHROW(CheckFieldExists(pars["Key"], "SubKey2"));

        REQUIRE_THROWS_MATCHES(
                    CheckFieldExists(pars["Key"], "NoSuchKey", "Context"),
                    InvalidInputError,
                    HasMessage<InvalidInputError>("Field 'NoSuchKey' (below 'Context') is not defined in the input"));
    }

    SECTION("FieldVal")
    {
        auto pars = json();
        REQUIRE_THROWS_AS(FieldVal(pars, "Key"), InvalidInputError);

        pars["Key"] = 0;
        REQUIRE_NOTHROW(FieldVal(pars, "Key"));
    }
}
