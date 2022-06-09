using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace ML.Utility.Logger
{
    public class MultiLog
    {
        private static readonly Lazy<MultiLog> lazy =
            new Lazy<MultiLog>(() => new MultiLog());

        public static Dictionary<int, RichLog[]> TextBoxManager =
            new Dictionary<int, RichLog[]>();

        public static MultiLog Instance => lazy.Value;

        public static MultiLog GetInsance()
        {
            return Instance;
        }


        public static void RegisterLog(int mainID, RichLog[] richLogs)
        {
            if (TextBoxManager.Keys.Contains(mainID)) return;
            TextBoxManager[mainID] = richLogs;
        }

        public static void UnRegisterLog(int mainID)
        {
            if (!TextBoxManager.Keys.Contains(mainID)) return;
            TextBoxManager.Remove(mainID);
        }

        public static void AddLog(int mainID, LogType logtype = LogType.Message, object msg = null)
        {
            if (!TextBoxManager.Keys.Contains(mainID)) return;
            var richLogs = TextBoxManager[mainID]
                .Where(a => a != null)
                .ToList();

            richLogs.AsParallel()
                .ForAll(r => { r.Print(logtype, msg); });
        }

        public static void AddLog(LogType logtype = LogType.Message, object msg = null)
        {
            var mainID = Thread.CurrentThread.ManagedThreadId;
            if (!TextBoxManager.Keys.Contains(mainID)) return;
            var richLogs = TextBoxManager[mainID]
                .Where(a => a != null)
                .ToList();

            richLogs.AsParallel()
                .ForAll(r => { r.Print(logtype, msg); });
        }
    }
}