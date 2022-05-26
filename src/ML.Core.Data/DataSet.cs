using System;
using NumSharp;

namespace ML.Core.Data
{
    public abstract class DataSet
    {
        public Tuple<NDArray, NDArray> Data { set; get; }

        public Tuple<DataSet, DataSet> Split()
        {
            return new Tuple<DataSet, DataSet>(this, this);
        }

        public DataSet Shuffle()
        {
            return this;
        }

        public static DataSet LoadFromTxtFile(
            string filename,
            HeadType headType = HeadType.NoHeader,
            string split = ",")
        {
            return new ClassifyDataSet();
        }
    }

    public class ClassifyDataSet : DataSet
    {
    }

    public class RegressDataSet : DataSet
    {
    }

    public enum HeadType
    {
        HasHeader,
        NoHeader
    }
}