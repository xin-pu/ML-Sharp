using AutoDiff;
using ML.Core.Losses;
using Numpy;
using Xunit;
using Xunit.Abstractions;

namespace ML.Core.Test
{
    public class LossTest : AbstractTest
    {
        public LossTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }


        [Fact]
        public void TestLeastSquares()
        {
            var x = np.ones(2, 2);
            var y = np.ones(2, 1);
            var weight = np.random.rand(2);

            var leastSquares = new LeastSquares(Regularization.None);
            leastSquares.Complie(new[] {new Variable(), new Variable()});
            var (gradient, loss) = leastSquares.Call(weight, x, y);


            print(loss);
            print(gradient);
        }

        [Fact]
        public void TestLeastAbsolute()
        {
            var x = np.ones(2, 2);
            var y = np.ones(2, 1);
            var weight = np.random.rand(2);

            var leastSquares = new LeastAbsolute(Regularization.None);
            leastSquares.Complie(new[] {new Variable(), new Variable()});
            var (gradient, loss) = leastSquares.Call(weight, x, y);


            print(loss);
            print(gradient);
        }
    }
}