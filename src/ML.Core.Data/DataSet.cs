﻿using System;

namespace ML.Core.Data
{
    public abstract class DataSet
    {
        public Tuple<DataSet, DataSet> Split()
        {
            return new Tuple<DataSet, DataSet>(this, this);
        }

        public DataSet Shuffle()
        {
            return this;
        }

        public static DataSet LoadFromTxtFile(string filename)
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
}