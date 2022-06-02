using MvvmCross.ViewModels;

namespace ML.Core.Trainers
{
    public class TrainConfig : MvxViewModel
    {
        private int _batchSize;
        private int _epoch;

        public int Epoch
        {
            get => _epoch;
            set => SetProperty(ref _epoch, value);
        }

        public int BatchSize
        {
            get => _batchSize;
            set => SetProperty(ref _batchSize, value);
        }

        public void Check()
        {
        }
    }
}