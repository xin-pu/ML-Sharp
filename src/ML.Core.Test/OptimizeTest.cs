using System;
using System.Linq;
using ML.Core.Optimizers;
using Numpy;
using Xunit;
using Xunit.Abstractions;

namespace ML.Core.Test
{
    public class OptimizeTest : AbstractTest
    {
        public OptimizeTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
            Weight = np.array(2.0, 1.0);
            Grad = a => np.array(0.1, 0.1);
        }


        protected NDarray Weight { set; get; }
        protected Func<NDarray, NDarray> Grad { set; get; }

        [Fact]
        public void TestMomentum()
        {
            var momentummo = new Momentum(1E-1);
            var weight = Weight.copy();

            Enumerable.Range(0, 2).ToList().ForEach(_ =>
            {
                weight = momentummo.Call(weight, Grad, 0);
                print(weight);
            });
        }

        [Fact]
        public void TestNesterov()
        {
            var nesterov = new Nesterov(1E-1);
            var weight = Weight.copy();

            Enumerable.Range(0, 2).ToList().ForEach(_ =>
            {
                weight = nesterov.Call(weight, Grad, 0);
                print(weight);
            });
        }

        [Fact]
        public void TestSGD()
        {
            var sgd = new SGD(1E-1);
            var weight = Weight.copy();
            Enumerable.Range(0, 2).ToList().ForEach(_ =>
            {
                weight = sgd.Call(weight, Grad, 0);
                print(weight);
            });
        }

        [Fact]
        public void TestAdam()
        {
            var adam = new Adam(1E-1);
            var weight = Weight.copy();

            Enumerable.Range(0, 2).ToList().ForEach(_ =>
            {
                weight = adam.Call(weight, Grad, 0);
                print(weight);
            });
        }

        [Fact]
        public void TestAdaDelta()
        {
            var adaDelta = new AdaDelta(1E-1);
            var weight = Weight.copy();

            Enumerable.Range(0, 2).ToList().ForEach(_ =>
            {
                weight = adaDelta.Call(weight, Grad, 0);
                print(weight);
            });
        }

        [Fact]
        public void TestRMSProp()
        {
            var rmsProp = new RMSProp(1E-1);
            var weight = Weight.copy();

            Enumerable.Range(0, 2).ToList().ForEach(_ =>
            {
                weight = rmsProp.Call(weight, Grad, 0);
                print(weight);
            });
        }
    }
}