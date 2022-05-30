using FluentAssertions;
using ML.Core.Transform;
using Numpy;
using Xunit;
using Xunit.Abstractions;

namespace ML.Core.Test
{
    public class TransformTest : AbstractTest
    {
        public TransformTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
            InputOneDim = np.array(new double[,] {{1}, {2}, {3}});
            InputMultiDim = np.array(new double[,] {{1, 1}, {2, 2}, {3, 3}});
        }

        protected NDarray InputOneDim { set; get; }
        protected NDarray InputMultiDim { set; get; }

        #region Kernel

        [Fact]
        public void PolyKernel()
        {
            var kernel = new Poly(1);
            var res = kernel.Call(InputMultiDim);

            var trueRes = np.array(new double[,]
            {
                {2, 4, 6},
                {4, 8, 12},
                {6, 12, 18}
            });
            np.array_equal(res, trueRes).Should().BeTrue();
        }

        [Fact]
        public void GaussianKernel()
        {
            var kernel = new Gaussian(2);
            var res = kernel.Call(InputMultiDim);
            print(res);
        }

        [Fact]
        public void LaprasKernel()
        {
            var kernel = new Lapras(2);
            var res = kernel.Call(InputMultiDim);
            print(res);
        }

        [Fact]
        public void SigmoidKernel()
        {
            var kernel = new Sigmoid(2, -2);
            var res = kernel.Call(InputMultiDim);
            print(res);
        }

        #endregion


        #region Normal

        [Fact]
        public void LineFirstOrder()
        {
            var tran = new LinearFirstorder();
            var res = tran.Call(InputOneDim);
            print(res);
            var trueRes = np.array(new double[,]
            {
                {1, 1},
                {1, 2},
                {1, 3}
            });
            np.array_equal(res, trueRes).Should().BeTrue();
        }


        [Fact]
        public void Polynomial()
        {
            var tran = new Polynomial(3);
            var res = tran.Call(InputOneDim);

            var trueRes = np.array(new double[,]
            {
                {1, 1, 1, 1},
                {1, 2, 4, 8},
                {1, 3, 9, 27}
            });
            np.array_equal(res, trueRes).Should().BeTrue();
        }

        [Fact]
        public void TrianglePoly()
        {
            var tran = new TrianglePoly(2);
            var res = tran.Call(InputOneDim);
            print(res);
        }

        #endregion
    }
}