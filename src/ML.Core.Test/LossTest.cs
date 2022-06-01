using System.Linq;
using AutoDiff;
using FluentAssertions;
using ML.Core.Losses;
using ML.Utilty;
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

        public NDarray getYPred()
        {
            var weights = Result.GetData<double>();
            var variables = new[] {new Variable(), new Variable()};
            var y_pred_array = Call(variables, X).Select(t => t.Evaluate(variables, weights))
                .ToArray();
            var y_pred = np.array(y_pred_array);
            return y_pred;
        }

        [Fact]
        public void TestLeastSquaresLoss()
        {
            var y_pred = np.array(1, 2, 3, 4);
            var y_true = np.array(1, 2, 2, 4);
            var loss = new LeastSquares().Call(y_pred, y_true);
            loss.Should().Be(0.125);
        }

        [Fact]
        public void TestLeastAbsoluteLoss()
        {
            var y_pred = np.array(1, 2, 3, 4);
            var y_true = np.array(1, 2, 2, 4);
            var loss = new LeastAbsolute().Call(y_pred, y_true);
            loss.Should().Be(0.25);
        }

        [Fact]
        public void TestBinaryLeastSquaresLoss()
        {
            var y_pred = np.array(-18.6, 0.51, 2.94, -12.8);
            var y_true = np.array(0, 1, 0, 0);
            var loss = new BinaryLeastSquares().Call(y_pred, y_true);
            print(loss);
        }

        [Fact]
        public void TestBinaryCrossEntropyLoss()
        {
            var y_pred = np.array(-18.6, 0.51, 2.94, -12.8);
            var y_true = np.array(0, 1, 0, 0);
            var loss = new BinaryCrossentropy().Call(y_pred, y_true);
            print(loss);
        }

        [Fact]
        public void TestBinarySoftmaxLoss()
        {
            var y_pred = np.array(-18.6, 0.51, 2.94, -12.8);
            var y_true = np.array(-1, 1, -1, -1);
            var loss = new BinarySoftmax().Call(y_pred, y_true);
            print(loss);
        }


        //[Fact]
        //public void TestLeastSquares2()
        //{
        //    var weights = Result.GetData<double>();
        //    var variables = new[] {new Variable(), new Variable()};


        //    var y_pred = Call(variables, X);


        //    var leastSquares = new LeastSquares();
        //    leastSquares.Complie(variables);
        //    var lossTerm = leastSquares.Call(y_pred, Y);

        //    var loss = lossTerm.Evaluate(variables, weights);
        //    var gradient = lossTerm.Differentiate(variables, weights);
        //    loss.Should().Be(0);
        //    gradient.Should().BeEquivalentTo(new double[] {0, 0});
        //}


        //[Fact]
        //public void TestLeastAbsolute()
        //{
        //    var weights = Result.GetData<double>();


        //    var variables = new[] {new Variable(), new Variable()};
        //    var y_pred = Call(variables, X);

        //    var leastSquares = new LeastAbsolute(0.1, Regularization.L2);
        //    leastSquares.Complie(variables);
        //    var lossTerm = leastSquares.Call(y_pred, Y);

        //    weights = weights.Select(a => a + 1E-7).ToArray();

        //    var loss = lossTerm.Evaluate(variables, weights);
        //    var gradient = lossTerm.Differentiate(variables, weights);
        //    print(loss);
        //    print(np.array(gradient));
        //}
    }
}