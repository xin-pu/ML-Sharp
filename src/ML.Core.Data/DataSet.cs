using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using FluentAssertions;
using MathNet.Numerics.Random;

namespace ML.Core.Data
{
    /// <summary>
    ///     数据集
    /// </summary>
    /// <typeparam name="T"></typeparam>
    [Serializable]
    public sealed class DataSet<T> : ICloneable
    {
        public DataSet(IList<T> dataList)
        {
            DataList = dataList;
        }

        public Type Type => typeof(T);

        public IList<T> DataList { get; }

        public int Count => DataList.Count;

        public object Clone()
        {
            var BF = new BinaryFormatter();
            var memStream = new MemoryStream();
            BF.Serialize(memStream, this);
            memStream.Flush();
            memStream.Position = 0;
            return BF.Deserialize(memStream);
        }

        /// <summary>
        ///     Shuffle
        /// </summary>
        /// <returns></returns>
        public DataSet<T> Shuffle()
        {
            var randomSource = SystemRandomSource.Default;
            var cache = new T[Count];
            DataList.CopyTo(cache, 0);
            var list = cache.ToList();
            var Returncache = new List<T>();
            while (list.Count > 0)
            {
                var currentIndex = randomSource.Next(0, list.Count);
                Returncache.Add(list[currentIndex]);
                list.RemoveAt(currentIndex);
            }

            return new DataSet<T>(Returncache);
        }

        public (DataSet<T>, DataSet<T>) Split(double per)
        {
            per.Should().BeInRange(0, 1, "per shoule be in range[0%,100%]");

            var shuffle = Shuffle();
            var trainCount = (int) Math.Round(Count * per, MidpointRounding.AwayFromZero);
            var valCount = Count - trainCount;
            return (shuffle.Take(trainCount), shuffle.Take(valCount));
        }

        /// <summary>
        ///     Order by Function
        /// </summary>
        /// <typeparam name="T2"></typeparam>
        /// <param name="a"></param>
        /// <returns></returns>
        public DataSet<T> Orderby<T2>(Func<T, T2> a)
        {
            var data = DataList.OrderBy(a).ToList();
            var dataset = new DataSet<T>(data);
            return dataset;
        }

        /// <summary>
        ///     Take function
        /// </summary>
        /// <param name="count"></param>
        /// <returns></returns>
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