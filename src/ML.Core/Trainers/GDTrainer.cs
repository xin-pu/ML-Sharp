using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using AutoDiff;
using FluentAssertions;
using GalaSoft.MvvmLight;
using ML.Core.Data;
using ML.Core.Losses;
using ML.Core.Metrics;
using ML.Core.Metrics.Regression;
using ML.Core.Models;
using ML.Core.Optimizers;
using Numpy;

namespace ML.Core.Trainers
{
    public class GDTrainer : ViewModelBase
    {
        private int _currentBatchIndex;
        private int _currentEpoch;


        private Loss _loss;
        private ObservableCollection<Metric> _metrics;
        private IModelGD _modelGd;
        private Optimizer _optimizer;

        private Dataset<DataView> _trainDataset;
        private TrainPlan _trainPlan;
        private Dataset<DataView> _valDataset;


        public Action<string> Print;

        public GDTrainer()
        {
            ModelGd = new MultipleLinearRegression();
            Optimizer = new Adam();
            Loss = new MeanSquared();
            Metrics = new ObservableCollection<Metric> {new MeanSquaredError(), new RSquared()};
            TrainPlan = new TrainPlan();
        }


        public IModelGD ModelGd
        {
            get => _modelGd;
            set => Set(ref _modelGd, value);
        }

        public Dataset<DataView> TrainDataset
        {
            get => _trainDataset;
            set => Set(ref _trainDataset, value);
        }

        public Dataset<DataView> ValDataset
        {
            get => _valDataset;
            set => Set(ref _valDataset, value);
        }

        public Optimizer Optimizer
        {
            get => _optimizer;
            set => Set(ref _optimizer, value);
        }

        public Loss Loss
        {
            get => _loss;
            set => Set(ref _loss, value);
        }

        public ObservableCollection<Metric> Metrics
        {
            get => _metrics;
            set => Set(ref _metrics, value);
        }

        public TrainPlan TrainPlan
        {
            get => _trainPlan;
            set => Set(ref _trainPlan, value);
        }

        public int CurrentEpoch
        {
            get => _currentEpoch;
            set => Set(ref _currentEpoch, value);
        }

        public int CurrentBatchIndex
        {
            get => _currentBatchIndex;
            set => Set(ref _currentBatchIndex, value);
        }


        public async Task Fit(CancellationTokenSource cancellation = null)
        {
            CurrentEpoch = 0;
            CurrentBatchIndex = 0;
            TrainDataset.Should().NotBeNull("dataset should not ne null");

            ModelGd.PipelineDataSet(TrainDataset);

            foreach (var e in Enumerable.Range(1, TrainPlan.Epoch))
            {
                CurrentEpoch = e;
                if (cancellation?.IsCancellationRequested == true)
                    return;

                await Task.Delay(TimeSpan.FromMilliseconds(10));

                var iEnumerator = TrainDataset.GetEnumerator(TrainPlan.BatchSize);

                while (iEnumerator.MoveNext() &&
                       iEnumerator.Current is Dataset<DataView> data)
                {
                    if (cancellation?.IsCancellationRequested == true)
                        return;

                    if (data.Count == 0)
                        continue;

                    var batchdataSet = data.ToDatasetNDarray();

                    var predTerms = ModelGd.CallGraph(batchdataSet.Feature);
                    var lossTerm = Loss.GetLossTerm(predTerms, batchdataSet.Label, ModelGd.Variables);

                    await Task.Delay(TimeSpan.FromMilliseconds(10));

                    NDarray GetGradient(NDarray weight)
                    {
                        var gradientArray = lossTerm.Differentiate(ModelGd.Variables, weight.GetData<double>());
                        var g = np.array(gradientArray);
                        return np.reshape(g, weight.shape);
                    }

                    ModelGd.Weights = Optimizer.Call(ModelGd.Weights.copy(), GetGradient, e - 1);
                }


                var trainMsg = new StringBuilder($"#{e:D4}\t");
                var train_loss = UpdateLossMetric(TrainDataset, true);
                trainMsg.Append($"[Loss]:{train_loss:F4}\t");

                if (ValDataset == null)
                    continue;
                if (ValDataset.Value == null)
                    continue;
                if (ValDataset.Value.Length == 0)
                    continue;

                var val_loss = UpdateLossMetric(ValDataset, false);
                trainMsg.Append($"\tVal\t[Loss]:{val_loss:F4}\t");
                foreach (var metric in Metrics) trainMsg.Append($"{metric}\t");


                Print?.Invoke(trainMsg.ToString());
            }
        }

        private double UpdateLossMetric(Dataset<DataView> dataset, bool isTrain)
        {
            var dataview = dataset.ToDatasetNDarray();
            var y_pred = ModelGd.Call(dataview.Feature);
            var y_true = dataview.Label;

            var loss = Loss.GetLoss(y_pred, y_true);
            Metrics.ToList().ForEach(m => m.Call(y_true, y_pred));

            return loss;
        }
    }
}