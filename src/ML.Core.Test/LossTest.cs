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
        private readonly NDarray result = np.array(1.1, 1.1);
        private readonly NDarray x = np.array(new double[,] {{1, 1}, {1, 2}});
        private readonly NDarray y_true = np.array(new double[,] {{2}, {3}});

        public LossTest(ITestOutputHelper testOutputHelper) : base(testOutputHelper)
        {
        }

        public Term[] Call(Variable[] weight, NDarray x)
        {
            return term.matmul(weight, x);
        }

        public NDarray getYPred()
        {
            var weights = result.GetData<double>();
            var variables = new[] {new Variable(), new Variable()};
            var y_pred_array = Call(variables, x)
                .Select(t => t.Evaluate(variables, weights))
                .ToArray();
            var y_pred = np.array(y_pred_array);
            return y_pred;
        }

        [Fact]
        public void TestLeastSquaresLoss()
        {
            var yPred = np.array(1, 2, 3, 4);
            var yTrue = np.array(1, 2, 2, 4);
            var loss = new LeastSquares().GetLoss(yPred, yTrue);
            loss.Should().Be(0.125);
        }

        [Fact]
        public void TestLeastAbsoluteLoss()
        {
            var yPred = np.array(1, 2, 3, 4);
            var yTrue = np.array(1, 2, 2, 4);
            var loss = new LeastAbsolute().GetLoss(yPred, yTrue);
            loss.Should().Be(0.25);
        }

        [Fact]
        public void TestBinaryLeastSquaresLoss()
        {
            var yPred = np.array(-18.6, 0.51, 2.94, -12.8);
            var yTrue = np.array(new double[] {0, 1, 0, 0});
            var loss = new BinaryLeastSquares().GetLoss(yPred, yTrue);
            print(loss);
        }

        [Fact]
        public void TestBinaryCrossEntropyLoss()
        {
            var yPred = np.array(-18.6, 0.51, 2.94, -12.8);
            var yTrue = np.array(new double[] {0, 1, 0, 0});
            var loss = new BinaryCrossentropy().GetLoss(yPred, yTrue);
            print(loss);
        }

        [Fact]
        public void TestBinarySoftmaxLoss()
        {
            var yPred = np.array(-18.6, 0.51, 2.94, -12.8);
            var yTrue = np.array(new double[] {-1, 1, -1, -1});
            var loss = new BinarySoftmax().GetLoss(yPred, yTrue);
            print(loss);
        }


        [Fact]
        public void TestLeastSquaresTerm()
        {
            var variables = new[] {new Variable(), new Variable()};
            var y_pred = Call(variables, x);

            var lossTerm = new LeastSquares().GetLossTerm(y_pred, y_true, variables);

            var loss = lossTerm.Evaluate(variables, result.GetData<double>());
            var gradient = lossTerm.Differentiate(variables, result.GetData<double>());
            print(loss);
            print(np.array(gradient));
        }


        [Fact]
        public void TestLeastAbsoluteTerm()
        {
            var variables = new[] {new Variable(), new Variable()};
            var y_pred = Call(variables, x);

            var leastSquares = new LeastAbsolute();

            var lossTerm = leastSquares.GetLossTerm(y_pred, y_true, variables);

            var loss = lossTerm.Evaluate(variables, result.GetData<double>());
            var gradient = lossTerm.Differentiate(variables, result.GetData<double>());

            print(loss);
            print(np.array(gradient));
        }
    }
}