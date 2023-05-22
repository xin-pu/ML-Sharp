using ML.Utility;
using Numpy;

namespace ML.Core.Models.NeuralNets
{
    public class Linear : Module
    {
        public override string Description { get; } = "y=xA \r\nT\r\n +b";

        private int _inFeatures;
        private int _outFeatures;
        private bool _withBias;

        public Linear(int inFeatures, int outFeatures, Dtype dtype, bool bias = true) : base(dtype)
        {
            InFeatures = inFeatures;
            OutFeatures = outFeatures;
            WithBias = bias;
        }

        public int InFeatures
        {
            protected set => SetProperty(ref _inFeatures, value);
            get => _inFeatures;
        }

        public int OutFeatures
        {
            protected set => SetProperty(ref _outFeatures, value);
            get => _outFeatures;
        }

        public bool WithBias
        {
            protected set => SetProperty(ref _withBias, value);
            get => _withBias;
        }

        public VariableMatrix? VariableWeights { set; get; }
        public VariableMatrix? VariableBias { set; get; }
        public NDarray? Weights { set; get; }
        public NDarray? Bias { set; get; }


        public override void PipeLineForWeights(NDarray input)
        {
            var k = (float) Math.Sqrt(InFeatures);

            VariableWeights = new VariableMatrix(OutFeatures, InFeatures);
            Weights = np.random.uniform(np.array(-k), np.array(k), new[] {OutFeatures, InFeatures});

            if (WithBias)
            {
                VariableBias = new VariableMatrix(OutFeatures, 1);
                Bias = np.random.uniform(np.array(-k), np.array(k), new[] {OutFeatures, 1});
            }
        }

        public override TermMatrix CallGraph(NDarray features)
        {
            var outTerm = new TermMatrix(1, OutFeatures);
            foreach (var c in Enumerable.Range(0, VariableWeights.Width))
            {
                var col = VariableWeights.GetColumn(c);
                outTerm[0, c] = TermOp.MatmulRow(col, features[$"...,{c}"]);
            }

            return outTerm;
        }

        public override NDarray Call(NDarray features)
        {
            var res = features * Weights;
            return WithBias ? res + Bias : res;
        }
    }
}