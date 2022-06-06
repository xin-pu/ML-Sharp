using AutoDiff;
using ML.Core.Data;
using ML.Utility;
using Numpy;

namespace ML.Core.Models
{
    public interface IModelGD<T> : IModel
        where T : DataView
    {
        Variable[] Variables { set; get; }

        void PipelineDataSet(Dataset<T> dataset);

        /// <summary>
        /// </summary>
        /// <param name="features">[batch size, variables]</param>
        /// <returns></returns>
        TermMatrix CallGraph(NDarray features);
    }
}