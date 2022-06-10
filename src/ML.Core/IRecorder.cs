using System;

namespace ML.Core
{
    public interface IRecorder
    {
        public string Name { get; }
        public Action<double> ReportToRecorder { set; get; }
    }
}