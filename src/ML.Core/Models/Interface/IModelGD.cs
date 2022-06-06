using AutoDiff;
using ML.Core.Data;
using Numpy;

namespace ML.Core.Models
{
    public interface IModelGD<T> : IModel
        where T : DataView
    {
        Variable[] Variables { set; get; }

        void PipelineDataSet(Dataset<T> dataset);

        Term[] CallGraph(NDarray features);
    }
}