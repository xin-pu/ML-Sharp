using System;
using Numpy;

namespace ML.Core.Data
{
    public class DatasetNDarray : IDisposable
    {
        public NDarray Feature;
        public NDarray Label;


        public void Dispose()
        {
            Feature = null;
            Label = null;
        }
    }
}