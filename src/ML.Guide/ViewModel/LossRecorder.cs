using System.Linq;
using GalaSoft.MvvmLight;
using LiveCharts;
using LiveCharts.Wpf;
using ML.Core;
using ML.Core.Trainers;

namespace ML.Guide.ViewModel
{
    public class LossRecorder : ViewModelBase
    {
        private SeriesCollection _series;

        public LossRecorder()
        {
            Series = new SeriesCollection();
        }

        public SeriesCollection Series
        {
            get => _series;
            set => Set(ref _series, value);
        }

        public void RegiserModel(GDTrainer gdTrainer)
        {
            var lossRecord = new Recorder(gdTrainer.Loss);
            var vallossRecord = new Recorder(gdTrainer.Loss);
            Series = new SeriesCollection
            {
                new StackedAreaSeries
                {
                    Values = lossRecord.Values,
                    Name = lossRecord.Name
                },
                new StackedAreaSeries
                {
                    Values = vallossRecord.Values,
                    Name = vallossRecord.Name
                }
            };
        }
    }

    public class MetricRecorder : ViewModelBase
    {
        private SeriesCollection _series;

        public MetricRecorder()
        {
            Series = new SeriesCollection();
        }

        public SeriesCollection Series
        {
            get => _series;
            set => Set(ref _series, value);
        }

        public void RegiserModel(GDTrainer gdTrainer)
        {
            var mertrics = gdTrainer.Metrics;
            var series = mertrics
                .Select(m => new Recorder(m))
                .Select(m => new StackedAreaSeries
                {
                    Values = m.Values,
                    Name = m.Name
                }).ToArray();
            Series = new SeriesCollection();
            Series.AddRange(series);
        }
    }

    public class Recorder : ViewModelBase
    {
        private IRecorder _iRecorder;
        private ChartValues<double> _values;

        public Recorder(IRecorder recorder)
        {
            IRecorder = recorder;
            Name = recorder.Name;
            Values = new ChartValues<double>();


            recorder.ReportToRecorder = Append;
        }

        public string Name { protected set; get; }

        public IRecorder IRecorder
        {
            get => _iRecorder;
            set => Set(ref _iRecorder, value);
        }

        public ChartValues<double> Values
        {
            get => _values;
            set => Set(ref _values, value);
        }

        public void Append(double value)
        {
            Values.Add(value);
        }
    }
}