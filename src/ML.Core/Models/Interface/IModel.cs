using ML.Core.Data;
using Numpy;

namespace ML.Core.Models
{
    public interface IModel
    {
        NDarray Weights { set; get; }
        string WeightFile { get; }
        double[] GetWeightArray();
        NDarray Call(NDarray features);
        NDarray Call(DataView features);

        void Save(string filename);
    }
}