using Numpy;

namespace ML.Core.Data
{
    /// <summary>
    ///     包含特征和标签的数据集对象
    /// </summary>
    public class DatasetNDarray : IDisposable
    {
        public DatasetNDarray(NDarray feature, NDarray label)
        {
            Feature = feature;
            Label = label;
        }

        public NDarray Feature { set; get; }
        public NDarray Label { set; get; }

        public void Dispose()
        {
            Feature.Dispose();
            Label.Dispose();
        }
    }
}