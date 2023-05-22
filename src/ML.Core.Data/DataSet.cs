using System.Collections;
using System.Text;
using CommunityToolkit.Mvvm.ComponentModel;
using FluentAssertions;
using Numpy;
using YAXLib;

namespace ML.Core.Data
{
    /// <summary>
    ///     数据集
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public sealed class Dataset<T> : ObservableObject, ICloneable
        where T : DataView
    {
        public Dataset(T[] value)
        {
            Value = value;
        }

        public Dataset(IEnumerable<T> value)
        {
            Value = value.ToArray();
        }

        public Type Type => typeof(T);

        public T[] Value { internal set; get; }

        public int Count => Value.Length;


        public override string ToString()
        {
            var str = new StringBuilder();
            str.AppendLine($"{Type.Name}({Count})");
            str.AppendLine(string.Join("\r\n", Value.ToList()));
            return str.ToString();
        }


        #region Iteration

        public int startingPoint;

        public IEnumerator GetEnumerator(int batchSize = 4)
        {
            if (batchSize <= 0)
                batchSize = Count;
            for (var index = 0; index <= Count / batchSize; index++)
            {
                var skip = batchSize * index;
                var it = Value
                    .Skip(skip)
                    .Take(batchSize)
                    .ToArray();
                yield return new Dataset<T>(it);
            }
        }

        public DatasetNDarray ToDatasetNDarray()
        {
            var arrays = Value.Select(a => a.ToDatasetNDarray()).ToList();
            var feature = arrays.Select(a => a.Feature).ToArray();
            var labels = arrays.Select(a => a.Label).ToArray();
            return new DatasetNDarray(np.vstack(feature), np.vstack(labels));
        }

        public NDarray ToFeatureNDarray()
        {
            var arrays = Value
                .Select(a => np.hstack(a.GetFeatureArray(), a.GetLabelArray()))
                .ToArray();
            return np.vstack(arrays);
        }

        #endregion


        #region operator

        /// <summary>
        ///     Clone
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            var yaxSerializer = new YAXSerializer<T[]>();
            var mem = yaxSerializer.Serialize(Value);
            var result = yaxSerializer.Deserialize(mem);
            var final = result == null
                ? new Dataset<T>(Array.Empty<T>())
                : new Dataset<T>(result);
            return final;
        }

        /// <summary>
        ///     Shuffle
        /// </summary>
        /// <returns></returns>
        public Dataset<T> Shuffle()
        {
            var randomSource = new Random();
            var cache = new T[Count];
            Value.CopyTo(cache, 0);
            var list = cache.ToList();
            var Returncache = new List<T>();
            while (list.Count > 0)
            {
                var currentIndex = randomSource.Next(0, list.Count);
                Returncache.Add(list[currentIndex]);
                list.RemoveAt(currentIndex);
            }

            return new Dataset<T>(Returncache.ToArray());
        }

        /// <summary>
        ///     Split function
        /// </summary>
        /// <param name="per"></param>
        /// <returns></returns>
        public (Dataset<T>, Dataset<T>) Split(double percentage)
        {
            percentage.Should().BeInRange(0, 1, "per shoule be in range[0%,100%]");

            var shuffle = Shuffle();
            var trainCount = (int) Math.Round(Count * percentage, MidpointRounding.AwayFromZero);
            var valCount = Count - trainCount;
            return (shuffle.Take(trainCount), shuffle.Take(valCount));
        }

        /// <summary>
        ///     Order by Function
        /// </summary>
        /// <typeparam name="T2"></typeparam>
        /// <param name="a"></param>
        /// <returns></returns>
        public Dataset<T> Orderby<T2>(Func<T, T2> a)
        {
            var data = Value.OrderBy(a).ToArray();
            var dataset = new Dataset<T>(data);
            return dataset;
        }

        /// <summary>
        ///     Take function
        /// </summary>
        /// <param name="count"></param>
        /// <returns></returns>
        public Dataset<T> Take(int count)
        {
            count.Should().BeGreaterOrEqualTo(0);
            count.Should().BeLessOrEqualTo(Count);

            var data = Value.Take(count).ToArray();
            return new Dataset<T>(data);
        }

        /// <summary>
        ///     Repeat function
        /// </summary>
        /// <param name="repeat"></param>
        /// <returns></returns>
        public Dataset<T> Repeat(int repeat)
        {
            var all = Enumerable.Range(0, repeat)
                .SelectMany(_ => ((Dataset<T>) Clone()).Value)
                .ToArray();
            return new Dataset<T>(all);
        }

        /// <summary>
        ///     Concat function
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public Dataset<T> Concat(Dataset<T> input)
        {
            var concatDataset = Value.Concat(input.Value)
                .ToArray();
            return new Dataset<T>(concatDataset);
        }

        #endregion
    }
}