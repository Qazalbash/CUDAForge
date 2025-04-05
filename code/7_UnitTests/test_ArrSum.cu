#include <gtest/gtest.h>

int ArrSum(const int32_t nElem);

const auto sizes = ::testing::Values(1 << 10, 1 << 8, 1 << 11, 1 << 12);

class ArrSumTest : public ::testing::TestWithParam<int> {};

INSTANTIATE_TEST_CASE_P(, ArrSumTest, sizes);

TEST_P(ArrSumTest, ) {
    const int size = GetParam();
    EXPECT_TRUE(ArrSum(size));
}
