namespace ML.Core.Transform
{
    public abstract class Kernel : Transform
    {
        protected Kernel(KernelType kernelType = KernelType.Gauss)
        {
            KernelType = kernelType;
        }

        public KernelType KernelType { protected set; get; }
    }

    public enum KernelType
    {
        Lapras,
        Poly,
        Gauss,
        Sigmoid
    }
}