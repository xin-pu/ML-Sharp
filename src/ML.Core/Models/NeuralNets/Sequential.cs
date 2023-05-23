using Numpy;

namespace ML.Core.Models.NeuralNets
{
    public class Sequential : Layer
    {
        public Sequential(params Layer[] layers)
        {
            Layers = layers;
        }

        public Layer[] Layers { protected set; get; }
        public NDarray NetActivation { set; get; } = np.empty();

        public override NDarray Forward(NDarray input)
        {
            NetActivation = input;

            foreach (var layer in Layers)
                NetActivation = layer.Forward(NetActivation);

            return NetActivation;
        }

        public override NDarray Backward(NDarray error)
        {
            throw new NotImplementedException();
        }
    }
}