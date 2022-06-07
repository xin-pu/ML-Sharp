using ML.Core.Data;
using Numpy;

namespace ML.Core.Models
{
    public interface IModel<in T>
        where T : DataView
    {
        NDarray Weights { set; get; }
        string WeightFile { get; }
        double[] GetWeightArray();
        NDarray Call(NDarray features);
        NDarray Call(T features);

        void Save(string filename);

        IModel<T> Load(string filename);
    }
}