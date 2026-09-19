/*
 * Exponentially scaled modified Bessel I for the RF hot-plasma kernel.
 *
 * This is the real, nonnegative-argument specialization of
 * petram.helper.bessel.ive.  RF lambda is nonnegative and harmonic order is
 * integral, so complex-argument continuation and non-integral order support
 * are intentionally outside this native ABI.
 */
#include <math.h>

#define BESSEL_EPSILON 1.0e-15
#define BESSEL_DIGITS 15.0
#define BESSEL_MIDDLE_SIZE 40
#define BESSEL_PI 3.14159265358979323846264338327950288

static double
ive_small(int order, double x)
{
    double term = 1.0;
    double sum = 0.0;

    for (int k = 0; k < 20; ++k) {
        const double factor = x * x / (4.0 * (k + 1) * (k + order + 1));
        sum += term;
        term *= factor;
        if (fabs(factor) < BESSEL_EPSILON) {
            break;
        }
    }
    return exp(-x) * sum * pow(x / 2.0, order) / tgamma(1.0 + order);
}

static double
ive_large(int order, double x)
{
    const double order_squared = 4.0 * order * order;
    double term = -(order_squared - 1.0) / (8.0 * x);
    double sum = 1.0;

    for (int i = 0; i < 15; ++i) {
        const int k = i + 2;
        sum += term;
        term = -term * (order_squared - (2.0 * k - 1.0) * (2.0 * k - 1.0)) /
               (k * 8.0 * x);
    }
    return sum / sqrt(2.0 * BESSEL_PI * x);
}

static double
ive_middle(int order, double x)
{
    double data[BESSEL_MIDDLE_SIZE] = {0.0};
    double normalization;

    data[BESSEL_MIDDLE_SIZE - 2] = 1.0;
    for (int i = 0; i < BESSEL_MIDDLE_SIZE - 2; ++i) {
        const int index = BESSEL_MIDDLE_SIZE - 3 - i;
        const double n = BESSEL_MIDDLE_SIZE - 2 - i;
        data[index] = 2.0 * n / x * data[index + 1] + data[index + 2];
    }

    normalization = data[0];
    for (int k = 1; k < BESSEL_MIDDLE_SIZE; ++k) {
        /* For integral order, nu is zero and this is exactly 2. */
        normalization += 2.0 * data[k];
    }

    return data[order] / normalization;
}

double
ive(int order, double x)
{
    const int n = order < 0 ? -order : order;

    if (x == 0.0) {
        return n == 0 ? 1.0 : 0.0;
    }
    if (x < 2.0 * sqrt(n + 1.0)) {
        return ive_small(n, x);
    }
    if (x > 1.2 * BESSEL_DIGITS + 2.4 && x > n * n / 2.0) {
        return ive_large(n, x);
    }
    if (n >= BESSEL_MIDDLE_SIZE) {
        return NAN;
    }
    return ive_middle(n, x);
}
