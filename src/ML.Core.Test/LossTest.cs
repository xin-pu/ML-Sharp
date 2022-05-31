using System.Linq;
using AutoDiff;
using FluentAssertions;
using ML.Core.Losses;
using Numpy;
using Xunit;
using Xunit.Abstractions;

namespace ML.Core.Test
{
    public class LossTest : AbstractTest
    {
        public NDarray Result = np.array(new double[] {1, 1});
        public NDarray X = np.array(new double[,] {{1, 1}, {1, 2}});
        public NDarray Y = np.array(new double[,] {{2}, {3}});

        public LossTest(ITestOutputHelper testOutputHelper) : base(testOutputHelper)
        {
        }

        public Term[] Call(Variable[] weight, NDarray x)
        {
            return term.matmul(weight, x);
        }

        [Fact]
        public void TestLeastSquares()
        {
            var weights = Result.GetData<double>();
            var variables = new[] {new Variable(), new Variable()};


            var y_pred = Call(variables, X);


            var leastSquares = new LeastSquares();
            leastSquares.Complie(variables);
            var lossTerm = leastSquares.Call(y_pred, Y);

            var loss = lossTerm.Evaluate(variables, weights);
            var gradient = lossTerm.Differentiate(variables, weights);
            loss.Should().Be(0);
            gradient.Should().BeEquivalentTo(new double[] {0, 0});
        }


        [Fact]
        public void TestLeastAbsolute()
        {
            var weights = Result.GetData<double>();


            var variables = new[] {new Variable(), new Variable()};
            var y_pred = Call(variables, X);

            var leastSquares = new LeastAbsolute(0.1, Regularization.L2);
            leastSquares.Complie(variables);
            var lossTerm = leastSquares.Call(y_pred, Y);

            weights = weights.Select(a => a + 1E-7).ToArray();

            var loss = lossTerm.Evaluate(variables, weights);
            var gradient = lossTerm.Differentiate(variables, weights);
            print(loss);
            print(np.array(gradient));
        }
    }
}