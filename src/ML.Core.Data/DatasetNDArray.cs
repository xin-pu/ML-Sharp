using Numpy;

namespace ML.Core.Data
{
    public class DatasetNDarray : IDisposable
    {
        public NDarray Feature { set; get; }
        public NDarray Label { set; get; }


        public void Dispose()
        {
            Feature = null;
            Label = null;
        }
    }
}