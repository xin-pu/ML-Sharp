using System.Linq;
using AutoDiff;
using FluentAssertions;
using ML.Core.Losses;
using ML.Utility;
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

        public Term[] Call(Variable[] weight)
        {
            return term.matmul(weight, x);
        }

        public NDarray getYPred()
        {
            var weights = result.GetData<double>();
            var variables = new[] {new Variable(), new Variable()};
            var y_pred_array = Call(variables)
                .Select(t => t.Evaluate(variables, weights))
                .ToArray();
            var y_pred = np.array(y_pred_array);
            return y_pred;
        }

        [Fact]
        public void TestLeastSquaresLoss()
        {
            var yTrue = np.array(new double[] {1, 2, 2, 4});
            var yPred = np.array(new double[] {1, 2, 3, 4});
            var loss = new MeanSquaredError().GetLoss(yPred, yTrue);
            loss.Should().Be(0.125);
        }

        [Fact]
        public void TestLeastAbsoluteLoss()
        {
            var yTrue = np.array(new double[] {1, 2, 2, 4});
            var yPred = np.array(new double[] {1, 2, 3, 4});
            var loss = new MeanAbsoluteError().GetLoss(yPred, yTrue);
            loss.Should().Be(0.25);
        }

        [Fact]
        public void TestLogCoshLoss()
        {
            var yTrue = np.array(new double[,] {{0, 1}, {0, 0}});
            var yPred = np.array(new double[,] {{1, 1}, {0, 0}});
            var loss = new LogCosh().GetLoss(yPred, yTrue);
            print(loss);
        }

        [Fact]
        public void TestBinaryLeastSquaresLoss()
        {
            var yTrue = np.array(new double[] {0, 0, 1, 0});

            var yPred = np.array(new double[] {0, 0, 1, 0});
            var loss1 = new BinaryLeastSquares().GetLoss(yPred, yTrue);
            loss1.Should().Be(0);

            yPred = np.array(-18.6, 0.51, 2.94, -12.8);
            var loss2 = new BinaryLeastSquares(LabelType.Logits).GetLoss(yPred, yTrue);
            print(loss2);
        }

        [Fact]
        public void TestBinaryCrossEntropyLoss()
        {
            var yTrue = np.array(new double[] {0, 0, 1, 0});

            var yPred = np.array(0.1, 0.1, 0.8, 0.1);
            var loss1 = new BinaryCrossentropy().GetLoss(yPred, yTrue);
            print(loss1);


            yPred = np.array(-18.6, 0.51, 2.94, -12.8);
            var loss2 = new BinaryLeastSquares(LabelType.Logits).GetLoss(yPred, yTrue);
            print(loss2);
        }

        [Fact]
        public void TestBinarySoftmaxLoss()
        {
            var yTrue = np.array(new double[] {-1, -1, 1, -1});

            var yPred = np.array(new double[] {-1, -1, 1, -1});
            var loss1 = new BinarySoftmax().GetLoss(yPred, yTrue);
            print(loss1);

            yPred = np.array(-18.6, 0.51, 2.94, -12.8);
            var loss2 = new BinarySoftmax(LabelType.Logits).GetLoss(yPred, yTrue);
            print(loss2);
        }


        [Fact]
        public void TestLeastSquaresTerm()
        {
            var variables = new[] {new Variable(), new Variable()};
            var y_pred = Call(variables);

            var lossTerm = new MeanSquaredError().GetLossTerm(y_pred, y_true, variables);

            var loss = lossTerm.Evaluate(variables, result.GetData<double>());
            var gradient = lossTerm.Differentiate(variables, result.GetData<double>());
            print(loss);
            print(np.array(gradient));
        }


        [Fact]
        public void TestLeastAbsoluteTerm()
        {
            var variables = new[] {new Variable(), new Variable()};
            var y_pred = Call(variables);

            var leastSquares = new MeanAbsoluteError();

            var lossTerm = leastSquares.GetLossTerm(y_pred, y_true, variables);

            var loss = lossTerm.Evaluate(variables, result.GetData<double>());
            var gradient = lossTerm.Differentiate(variables, result.GetData<double>());

            print(loss);
            print(np.array(gradient));
        }
    }
}