using System.Collections.Generic;
using System.IO;
using System.Linq;
using FluentAssertions;
using ML.Core.Data;
using ML.Core.Data.Loader;
using ML.Core.Test.DataStructs;
using Xunit;
using Xunit.Abstractions;

namespace ML.Core.Test
{
    public class DatasetTest : AbstractTest
    {
        private readonly string dataFolder = @"..\..\..\..\..\data";

        public DatasetTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }


        public Dataset<IrisData> GetIrisDataSet(int count)
        {
            var iris = new List<IrisData>();
            Enumerable.Range(0, count).ToList()
                .ForEach(_ => iris.Add(IrisData.RandomIris()));
            var dataset = new Dataset<IrisData>(iris.ToArray());
            return dataset;
        }

        [Fact]
        public void DataSetCreate()
        {
            var dataset = GetIrisDataSet(1);
            print(dataset);
        }

        [Fact]
        public void DataSetSplit()
        {
            var dataset = GetIrisDataSet(10);
            var res = dataset.Split(0.8);
            print(res.Item1);
            print(res.Item2);
        }

        [Fact]
        public void DataSetOrder()
        {
            var dataset = GetIrisDataSet(5);
            var orderDataSet = dataset.Orderby(c => c.SepalLength);
            print(orderDataSet);
        }

        [Fact]
        public void DataSetShuffle()
        {
            var dataset = GetIrisDataSet(10);
            var shuffleDataSet = dataset.Shuffle();
            print(dataset);
            print(shuffleDataSet);
        }


        [Fact]
        public void DataSetClone()
        {
            var dataset = GetIrisDataSet(1);
            var cloneDataSet = dataset.Clone();
            cloneDataSet.Should().NotBe(dataset);
            print(dataset);
            print(cloneDataSet);
        }

        [Fact]
        public void DataSetConcat()
        {
            var dataset1 = GetIrisDataSet(1);
            var dataset2 = GetIrisDataSet(1);
            var concatOne = dataset1.Concat(dataset2);
            print(concatOne);
        }


        [Fact]
        public void DataSetRepeat()
        {
            var dataset1 = GetIrisDataSet(1);
            var repeatOne = dataset1.Repeat(5);
            print(repeatOne);
        }

        [Fact]
        public void DataSetLoadIris()
        {
            var path = Path.Combine(dataFolder, "iris-train.txt");
            var dataset = TextLoader<IrisData>.LoadDataSet(path, splitChar: '\t');
            print(dataset);
        }

        [Fact]
        public void DataSetLoadMultiIris()
        {
            var pathes = new[]
            {
                Path.Combine(dataFolder, "iris-train.txt"),
                Path.Combine(dataFolder, "iris-test.txt")
            };
            var dataset = TextLoader<IrisData>.LoadDataSet(pathes, splitChar: '\t');
            print(dataset);
        }


        //[Fact]
        //public void DataSetLoadOptdigits()
        //{
        //    var path = Path.Combine(dataFolder, "optdigits-train.csv");
        //    var dataset = TextLoader<OptdigitData>.LoadDataSet(path, splitChar: ',');
        //    print(dataset);
        //}


        [Fact]
        public void DataSetToNDarray()
        {
            var dataset1 = GetIrisDataSet(10);
            var arrayND = dataset1.ToDatasetNDarray();
            print(arrayND.Label);
            print(arrayND.Feature);
        }

        [Fact]
        public void DataSetIteration()
        {
            var path = Path.Combine(dataFolder, "iris-train.txt");
            var dataset = TextLoader<IrisData>.LoadDataSet(path, splitChar: '\t');

            var i = dataset.GetEnumerator();
            while (true)
            {
                var item = i.MoveNext();
                if (item == false) break;
                if (i.Current is Dataset<IrisData> c) print(c);
            }
        }
    }
}