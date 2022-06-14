using System.Collections.Generic;
using System.IO;
using System.Linq;
using ML.Core.Data.DataStructs;
using ML.Core.Data.Loader;
using Numpy;
using Numpy.Models;
using Xunit;
using Xunit.Abstractions;

namespace ML.Core.Test
{
    public class KNNTest : AbstractTest
    {
        private readonly string dataFolder = @"..\..\..\..\..\data";

        public KNNTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void Test()
        {
            var path = Path.Combine(dataFolder, "iris-train.txt");
            var data = TextLoader.LoadDataSet<IrisData>(path, new[] {'\t'});
            var input = data.ToDatasetNDarray().Feature;

            var batchsize = input.shape[0];
            var temp_centroid_group = np.zeros(new Shape(batchsize), np.int64);
            var centroid_group = np.ones_like(temp_centroid_group);
            var kmeans = np.array(Enumerable.Range(0, 3).Select(i => input[np.random.choice(batchsize)]));

            var index = 0;

            print(temp_centroid_group - centroid_group);
            while (!np.all(np.equal(temp_centroid_group, centroid_group)))
            {
                index++;
                centroid_group = getDistance(input, kmeans);

                /// Update Kmeans
                kmeans = getKMeans(input, centroid_group);
                print(index);
            }


            print(centroid_group);
        }

        /// <summary>
        ///     计算每个点 最近
        /// </summary>
        /// <param name="input"></param>
        /// <param name="kmeans"></param>
        /// <returns></returns>
        public NDarray getDistance(NDarray input, NDarray kmeans)
        {
            var k = 3;

            var num_pts = input.shape[0];
            var num_feats = input.shape[1];

            np.tile(kmeans, np.array(num_pts, 1));
            var centroid_matrix = np.reshape(np.tile(kmeans, np.array(num_pts, 1)), new Shape(num_pts, k, num_feats));
            var point_matrix = np.reshape(np.tile(input, np.array(1, k)), new Shape(num_pts, k, num_feats));
            var dis = np.linalg.norm(point_matrix - centroid_matrix, axis: -1, ord: 2);
            return np.argmin(dis, -1);
        }

        public NDarray getKMeans(NDarray input, NDarray cluster)
        {
            var array = cluster.GetData<int>();
            var batch = cluster.shape[0];
            var k = 3;
            var dict = new Dictionary<int, List<NDarray>>();
            foreach (var i in Enumerable.Range(0, k))
            {
                var list = new List<NDarray>();
                foreach (var b in Enumerable.Range(0, batch))
                    if (array[b] == i)
                        list.Add(input[b]);

                dict[i] = list;
            }

            print(dict[0].Count);
            print(dict[1].Count);
            print(dict[2].Count);
            var kmeans = dict.Select(a =>
            {
                var arr = np.vstack(a.Value.ToArray());
                var s = np.mean(arr, 0);
                return s;
            }).ToArray();

            return np.vstack(kmeans);
            ;
        }
    }
}