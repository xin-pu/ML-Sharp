using System.Collections.Generic;
using System.Linq;
using FluentAssertions;
using ML.Core.Data;
using Numpy;
using Numpy.Models;

namespace ML.Core.Models
{
    public class KNN : IModel
    {
        public KNN(int k, bool regression = false)
        {
            K = k;
            Regression = regression;
        }

        public bool Regression { set; get; }
        public int K { get; set; }
        public NDarray Features { get; set; }

        public NDarray Labels { get; set; }
        public string Name => GetType().Name;
        public NDarray Weights { get; set; }
        public string WeightFile => $"{Name}.txt";

        public double[] GetWeightArray()
        {
            return null;
        }


        public NDarray Call(NDarray features)
        {
            features.ndim.Should().Be(Features.ndim);
            var res = new List<NDarray>();
            foreach (var index in Enumerable.Range(0, features.shape[0]))
            {
                var input = features[index];

                var dis = np.linalg.norm(input - Features, 2, -1, true);

                var knn = dis.GetData<double>()
                    .Select((v, i) => (i, v))
                    .OrderBy(p => p.v)
                    .Take(K)
                    .Select(p => p.i)
                    .ToList();

                var y_knn = Labels[np.array(knn.ToArray())];

                if (Regression)
                {
                    res.Add(np.average(y_knn, new Axis(0)));
                }
                else
                {
                    var y_knn_int = np.array(y_knn.GetData<int>(), np.int64);
                    res.Add(np.argmax(np.bincount(y_knn_int)));
                }
            }

            var newshape = new Shape(features.shape[0], Labels.shape[1]);
            return np.reshape(np.hstack(res.ToArray()), newshape);
        }


        public NDarray Call(DataView features)
        {
            var datas = new Dataset<DataView>(new[] {features});
            return Call(datas.ToDatasetNDarray().Feature)[0];
        }

        public void Save(string filename)
        {
        }

        public void LoadDataView(NDarray features, NDarray labels)
        {
            Features = features;
            Labels = labels;
        }
    }
}