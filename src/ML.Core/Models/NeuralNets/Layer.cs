using CommunityToolkit.Mvvm.ComponentModel;
using Numpy;

namespace ML.Core.Models.NeuralNets
{
    /// <typeparam name="T"></typeparam>
    public abstract class Layer : ObservableObject
    {
        public abstract NDarray Forward(NDarray input);
        public abstract NDarray Backward(NDarray error);
    }
}