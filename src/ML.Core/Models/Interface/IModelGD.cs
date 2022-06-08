using AutoDiff;
using ML.Core.Data;
using ML.Utility;
using Numpy;

namespace ML.Core.Models
{
    public interface IModelGD : IModel
    {
        Variable[] Variables { set; get; }

        void PipelineDataSet(Dataset<DataView> dataset);

        /// <summary>
        /// </summary>
        /// <param name="features">[batch size, variables]</param>
        /// <returns></returns>
        TermMatrix CallGraph(NDarray features);
    }
}