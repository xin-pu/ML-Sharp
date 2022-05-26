using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using FluentAssertions;

namespace ML.Core.Data
{
    /// <summary>
    ///     数据集
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public sealed class DataSet<T>
    {
        public DataSet(IList<T> dataList)
        {
            DataList = dataList;
            Type = typeof(T);
        }

        public Type Type { get; }

        public IList<T> DataList { set; get; }

        public int Count => DataList.Count;

        public DataSet<T> Shuffle()
        {
            return this;
        }

        public (DataSet<T>, DataSet<T>) Split(double per)
        {
            per.Should().BeInRange(0, 1, "per shoule be in range[0%,100%]");

            var shuffle = Shuffle();
            var trainCount = (int) Math.Round(Count * per, MidpointRounding.AwayFromZero);
            var valCount = Count - trainCount;
            return (shuffle.Take(trainCount), shuffle.Take(valCount));
        }

        public DataSet<T> Orderby<T2>(Func<T, T2> a)
        {
            var data = DataList.OrderBy(a).ToList();
            var dataset = new DataSet<T>(data);
            return dataset;
        }

        public DataSet<T> Take(int count)
        {
            count.Should().BePositive();
            count.Should().BeLessOrEqualTo(Count);

            var data = DataList.Take(count).ToList();
            return new DataSet<T>(data);
        }

        public override string ToString()
        {
            var str = new StringBuilder();
            str.AppendLine($"{Type.Name}({Count})");
            str.AppendLine(string.Join("\r\n", DataList));
            return str.ToString();
        }
    }
}