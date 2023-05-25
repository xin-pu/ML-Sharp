using ML.Core.Optimizers;
using Numpy;

namespace ML.Core.Models.NeuralNets
{
    /// <summary>
    ///     Activation Output=Input*Weights.T+Bias
    /// </summary>
    public class Linear : Layer
    {
        public Linear(
            int inFeatures,
            int outFeatures,
            bool withBias = true)
        {
            InFeatures = inFeatures;
            OutFeatures = outFeatures;
            WithBias = withBias;

            var bound = (float) Math.Sqrt(1f / InFeatures);
            Weights = np.random.uniform(np.array(-bound), np.array(bound), new[] {OutFeatures, InFeatures});

            Bias = WithBias
                ? np.random.uniform(np.array(-bound), np.array(bound), new[] {OutFeatures})
                : np.zeros(OutFeatures);
        }

        public NDarray Weights { set; get; }
        public NDarray Bias { set; get; }


        /// <summary>
        ///     误差项
        /// </summary>
        public NDarray Errs { set; get; } = np.empty();


        public int InFeatures { protected set; get; }
        public int OutFeatures { protected set; get; }
        public bool WithBias { protected set; get; }

        public void ResetParameters()
        {
            var bound = (float) Math.Sqrt(1f / InFeatures);
            Weights = np.random.uniform(np.array(-bound), np.array(bound), new[] {OutFeatures, InFeatures});
            Bias = WithBias
                ? np.random.uniform(np.array(-bound), np.array(bound), new[] {OutFeatures})
                : np.zeros(OutFeatures);
        }

        public override NDarray Forward(NDarray input)
        {
            var y = input.matmul(Weights.T);
            var res = Output = y + (WithBias ? Bias : np.zeros_like(y));
            return res;
        }

        public override NDarray Backward(NDarray gradient, Optimizer optimizer, int epoch = 0)
        {
            var perError = gradient.sum(0);
            Weights = optimizer.Call(Weights, gradient, epoch);
            return perError;
        }
    }
}