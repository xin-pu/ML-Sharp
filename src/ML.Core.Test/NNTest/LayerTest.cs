using ML.Core.Models.NeuralNets;
using Numpy;
using Xunit;
using Xunit.Abstractions;

namespace ML.Core.Test.NNTest
{
    public class LayerTest : AbstractTest
    {
        public LayerTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void LinearTest()
        {
            var linear = new Linear(20, 30);
            var input = np.random.rand(4, 20);
            var res = linear.Forward(input);
            print(res);
        }

        [Fact]
        public void ReluTest()
        {
            var reLu = new ReLU();
            var input = np.random.uniform(np.array(-1f), np.array(1f), new[] {4, 20});
            var res = reLu.Forward(input);
            print(res);
        }

        [Fact]
        public void SigmoidTest()
        {
            var reLu = new Sigmoid();
            var input = np.random.uniform(np.array(-1f), np.array(1f), new[] {4, 20});
            var res = reLu.Forward(input);
            print(res);
        }

        [Fact]
        public void SequentialTest()
        {
            var linear1 = new Linear(4, 30);
            var relu = new Sigmoid();
            var linear2 = new Linear(30, 4);

            var input = np.array(new float[,] {{1, 2, 3, 4}, {1, 2, 3, 5}});
            var sequential = new Sequential(linear1, relu, linear2);
            var res = sequential.Forward(input);
            print(res);
        }
    }
}