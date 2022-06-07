using FluentAssertions;
using Numpy;

namespace ML.Core.Metrics
{
    public abstract class Metric
    {
        public double ValueError { protected set; get; }

        public abstract string Describe { get; }

        public double Call(NDarray y_true, NDarray y_pred)
        {
            ValueError = call(y_true, y_pred);

            return ValueError;
        }

        internal virtual void precheck(NDarray y_true, NDarray y_pred)
        {
            y_true.shape.Should().BeEquivalentTo(y_pred.shape, "shape of y_true and y_pred should be same.");
        }

        internal abstract double call(NDarray y_true, NDarray y_pred);

        public override string ToString()
        {
            return $"{GetType().Name}:\t{ValueError:F4}";
        }
    }
}