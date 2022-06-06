using Numpy;

namespace ML.Core.Models
{
    public interface IModel
    {
        NDarray Weights { set; get; }
        double[] GetWeightArray();
        NDarray Call(NDarray features);
    }
}