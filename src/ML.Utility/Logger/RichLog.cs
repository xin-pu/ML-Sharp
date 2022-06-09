using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Markup;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ML.Utility.Logger
{
    public class RichLog
    {
        public RichLog(RichTextBox richTextBox)
        {
            RichTextBox = richTextBox;
            initialCallBack();
        }


        public RichTextBox RichTextBox { protected set; get; }

        private Dictionary<LogType, Color> ColorMap =>
            new Dictionary<LogType, Color>
            {
                [LogType.Message] = Colors.Black,
                [LogType.Debug] = Colors.Fuchsia,
                [LogType.Error] = Colors.Red,
                [LogType.Warning] = Colors.DarkOrange,
                [LogType.Mark] = Colors.SeaGreen
            };

        private void initialCallBack()
        {
            AddMsg = addMsg;
            AddLink = addLink;
            Addimage = addImage;
            AddTable = addTabel;
            AddSpecTable = addSpecTabel;
            AddDictionary = addDictionary;
            AddList = addList;
            Clearlog = clearLog;
            SaveRTF = saveRTF;
            SaveLog = saveLog;
        }


        public void Print(LogType logtype = LogType.Message, object msg = null)
        {
            try
            {
                switch (logtype)
                {
                    case LogType.Dict:
                        RichTextBox.Dispatcher?.Invoke(AddMsg, LogType.Dict, string.Empty);
                        RichTextBox.Dispatcher?.Invoke(AddDictionary, msg);
                        break;
                    case LogType.List:
                        RichTextBox.Dispatcher?.Invoke(AddMsg, LogType.List, string.Empty);
                        RichTextBox.Dispatcher?.Invoke(AddList, msg);
                        break;
                    case LogType.Table:
                        RichTextBox.Dispatcher?.Invoke(AddMsg, LogType.Table, string.Empty);
                        RichTextBox.Dispatcher?.Invoke(AddTable, msg);
                        break;
                    case LogType.Spec:
                        RichTextBox.Dispatcher?.Invoke(AddMsg, LogType.Spec, string.Empty);
                        RichTextBox.Dispatcher?.Invoke(AddSpecTable, msg);
                        break;
                    case LogType.Link:
                        RichTextBox.Dispatcher?.Invoke(AddMsg, LogType.Link, string.Empty);
                        RichTextBox.Dispatcher?.Invoke(AddLink, msg);
                        break;
                    case LogType.Image:
                        RichTextBox.Dispatcher?.Invoke(Addimage, msg);
                        break;
                    default:
                        RichTextBox.Dispatcher?.Invoke(AddMsg, logtype, msg?.ToString());
                        break;
                }
            }
            catch (Exception ex)
            {
                RichTextBox.Dispatcher?.Invoke(AddMsg, LogType.Error, ex.Message);
            }
        }

        public void SaveTxtLog(string path)
        {
            RichTextBox.Dispatcher?.Invoke(SaveLog, path);
        }

        public void SaveRtfLog(string path)
        {
            RichTextBox.Dispatcher?.Invoke(SaveRTF, path);
        }

        public void Clear()
        {
            RichTextBox.Dispatcher?.Invoke(Clearlog);
        }


        #region callback

        private Action<object> AddDictionary;
        private Action<string> Addimage;
        private Action<string> AddLink;
        private Action<object> AddList;
        private Action<LogType, string> AddMsg;
        private Action<object> AddSpecTable;
        private Action<object> AddTable;
        private Action Clearlog;
        private Action<string> SaveLog;
        private Action<string> SaveRTF;

        #endregion


        #region Operater

        private void clearLog()
        {
            RichTextBox.Document.Blocks.Clear();
        }

        private void saveLog(string path)
        {
            try
            {
                var rtb = RichTextBox;
                var textRange = new TextRange(
                    rtb.Document.ContentStart,
                    rtb.Document.ContentEnd
                );

                var sw = new StreamWriter(path);
                sw.Write(XamlWriter.Save(textRange.Text));
                sw.Flush();
                sw.Close();
            }
            catch (Exception ex)
            {
                addMsg(LogType.Error, $"Save Log Meet Exception:{ex.Message}");
            }
        }

        private void saveRTF(string path)
        {
            try
            {
                var rtb = RichTextBox;
                var textRange = new TextRange(
                    rtb.Document.ContentStart,
                    rtb.Document.ContentEnd
                );

                var file = new FileStream(path, FileMode.Create);
                textRange.Save(file, DataFormats.Rtf);
                file.Close();
            }
            catch (Exception ex)
            {
                addMsg(LogType.Error, $"Save Rtf Meet Exception:{ex.Message}");
            }
        }

        #endregion

        #region Add MSG

        /// <summary>
        ///     Log Message with different Level
        /// </summary>
        /// <param name="id"></param>
        /// <param name="logType"></param>
        /// <param name="msg"></param>
        private void addMsg(LogType logType, string msg)
        {
            if (logType == LogType.None) return;
            var para = new Paragraph
            {
                Margin = new Thickness(0),
                LineHeight = 1
            };
            switch (logType)
            {
                case LogType.NewLine:
                case LogType.End:
                    break;

                case LogType.Title1:
                case LogType.Title2:
                case LogType.Title3:
                    para = createTitleParagraph(logType, msg);
                    break;

                case LogType.Warning:
                case LogType.Error:
                    para = createParagraph(logType, msg);
                    break;
                default:
                    para = createParagraph(logType, msg);
                    break;
            }

            RichTextBox.Document.Blocks.Add(para);
        }

        private Paragraph createTitleParagraph(LogType logType, string title)
        {
            var r = new Run
            {
                Text = title
            };
            var para = new Paragraph
            {
                Margin = new Thickness(0),
                TextAlignment = TextAlignment.Center
            };
            switch (logType)
            {
                case LogType.Title1:
                    r.FontSize = 24;
                    r.FontWeight = FontWeights.Bold;
                    r.TextDecorations = TextDecorations.Underline;

                    para.LineHeight = 30;
                    break;

                case LogType.Title2:
                    r.FontSize = 18;
                    r.FontWeight = FontWeights.DemiBold;

                    para.LineHeight = 25;

                    break;
                case LogType.Title3:
                    r.FontSize = 16;
                    r.FontWeight = FontWeights.DemiBold;

                    para.LineHeight = 5;
                    break;
            }

            para.Inlines.Add(r);
            return para;
        }

        private Paragraph createParagraph(LogType logType, string message, bool bold = false)
        {
            var strBuild = new StringBuilder();
            strBuild.Append($"{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff}\t");
            strBuild.Append($"[{logType.ToString().ToUpper().Substring(0, 3)}]\t");
            strBuild.Append(message);
            var r = new Run
            {
                Text = strBuild.ToString(),
                FontSize = 11,
                FontWeight = bold ? FontWeights.Bold : FontWeights.Normal,
                Foreground = ColorMap.Keys.Contains(logType)
                    ? new SolidColorBrush(ColorMap[logType])
                    : new SolidColorBrush(ColorMap[LogType.Message])
            };
            var para = new Paragraph
            {
                Margin = new Thickness(0),
                LineHeight = 1
            };
            para.Inlines.Add(r);
            return para;
        }

        private void addLink(string fileFullPath)
        {
            var pParagrahp = new Paragraph
            {
                LineHeight = 1,
                Margin = new Thickness(0)
            };
            var pLinkRun = new Run(fileFullPath);
            var pHyperlink = new Hyperlink(pLinkRun)
            {
                NavigateUri = new Uri(fileFullPath, UriKind.Absolute)
            };
            pHyperlink.MouseLeftButtonDown += Hyperlink_Click;
            pHyperlink.MouseEnter += Hyperlink_MouseEnter;
            pParagrahp.Inlines.Add(pHyperlink);
            RichTextBox.Document.Blocks.Add(pParagrahp);
        }

        private void addImage(string imageSource)
        {
            try
            {
                var pParagrahp = new Paragraph
                {
                    LineHeight = 5,
                    Margin = new Thickness(0)
                };
                var c = new Figure
                {
                    Width = new FigureLength(400),
                    Padding = new Thickness(5),
                    WrapDirection = WrapDirection.None,
                    HorizontalAnchor = FigureHorizontalAnchor.ContentCenter
                };
                var image = new BlockUIContainer
                {
                    Child = new Image {Source = new BitmapImage(new Uri(imageSource))}
                };
                c.Blocks.Add(image);
                pParagrahp.Inlines.Add(c);
                RichTextBox.Document.Blocks.Add(pParagrahp);
            }
            catch (Exception ex)
            {
                addMsg(LogType.Error, $"Print Image Meet Exception:{ex.Message}");
            }
        }

        private static void Hyperlink_MouseEnter(object sender, MouseEventArgs e)
        {
            if (!(sender is Hyperlink pH)) return;
            pH.Cursor = Cursors.Hand;
        }

        private static void Hyperlink_Click(object sender, RoutedEventArgs e)
        {
            var pH = sender as Hyperlink;
            try
            {
                Process.Start("explorer.exe", pH?.NavigateUri.LocalPath);
            }
            catch
            {
                e.Handled = true;
            }
        }

        #endregion

        #region Add List

        private void addList(object list)
        {
            var type = list.GetType();
            if (type.FullName == null)
                return;
            if (type.FullName.Contains("Enumerable"))
            {
                var type1 = type.GenericTypeArguments[1];
                var typeThis = typeof(RichLog);
                var mi = typeThis.GetMethod("addEnumerableGenericity");
                var method = mi?.MakeGenericMethod(type1);
                method?.Invoke(this, new[] {list});
                return;
            }

            if (type.FullName.Contains("List"))
            {
                var type1 = type.GenericTypeArguments[0];
                var typeThis = typeof(RichLog);
                var mi = typeThis.GetMethod("addListGenericity");
                var method = mi?.MakeGenericMethod(type1);
                method?.Invoke(this, new[] {list});
            }
        }

        // ReSharper disable once UnusedMember.Local
        public void addEnumerableGenericity<T>(IEnumerable<T> list)
        {
            var k = new List
            {
                MarkerStyle = TextMarkerStyle.Box,
                LineHeight = 1
            };
            list.ToList().ForEach(a =>
            {
                var r = new Run {Text = $"\t{a}"};
                var para = new Paragraph
                {
                    LineHeight = 1,
                    Margin = new Thickness(0)
                };
                para.Inlines.Add(r);
                k.ListItems.Add(new ListItem(para));
            });
            RichTextBox.Document.Blocks.Add(k);
        }

        // ReSharper disable once UnusedMember.Local
        public void addListGenericity<T>(List<T> list)
        {
            var k = new List
            {
                MarkerStyle = TextMarkerStyle.Box,
                LineHeight = 1
            };
            list.ToList().ForEach(a =>
            {
                var r = new Run {Text = $"\t{a}"};
                var para = new Paragraph
                {
                    LineHeight = 1,
                    Margin = new Thickness(0)
                };
                para.Inlines.Add(r);
                k.ListItems.Add(new ListItem(para));
            });
            RichTextBox.Document.Blocks.Add(k);
        }

        #endregion

        #region Add Dict

        private void addDictionary(object dict)
        {
            var type = dict.GetType();

            if (type.FullName != null && type.FullName.Contains("Dictionary"))
            {
                var type1 = type.GenericTypeArguments[0];
                var type2 = type.GenericTypeArguments[1];
                var typeThis = typeof(RichLog);
                var mi = typeThis.GetMethod("addDictionaryGenericity");
                var method = mi?.MakeGenericMethod(type1, type2);
                method?.Invoke(this, new[] {dict});
            }
        }

        public void addDictionaryGenericity<T1, T2>(Dictionary<T1, T2> dict)
        {
            var list = dict?.ToList()
                .Select(pair => $"[{pair.Key}]:\t\t\t{pair.Value}")
                .ToList();
            RichTextBox.Dispatcher?.Invoke(AddList, list);
        }

        #endregion

        #region Add Table

        private void addTabel(object objectTable)
        {
            if (objectTable is IEnumerable<string> listtable)
                addTabel(listtable);
            else if (objectTable is DataTable table)
                addTabel(table);
            else
                throw new Exception($"Log Table Fail. Can't Support {objectTable.GetType()}");
        }

        private void addSpecTabel(object objectTable)
        {
            if (objectTable is IEnumerable<string> listtable)
                addTabel(listtable, true);
            else if (objectTable is DataTable table)
                addTabel(table);
            else
                throw new Exception($"Log Table Fail. Can't Support {objectTable.GetType()}");
        }

        private void addTabel(IEnumerable<string> table, bool isSpecTable = false)
        {
            var tableBlock = new Table();

            var rowGroup = new TableRowGroup();
            tableBlock.RowGroups.Add(rowGroup);

            var tableList = table.ToList();
            Enumerable.Range(0, tableList.Count).ToList().ForEach(index =>
            {
                var a = tableList[index];
                var tabelRow = new TableRow();
                var d = a.Split('\t');

                var color = isSpecTable
                    ? index == 0
                        ? new SolidColorBrush(Colors.Black)
                        : a.ToUpper().Contains("TRUE")
                            ? new SolidColorBrush(Colors.Green)
                            : new SolidColorBrush(Colors.Red)
                    : index == 0
                        ? new SolidColorBrush(Colors.Black)
                        : new SolidColorBrush(Colors.DarkBlue);

                d.ToList().ForEach(str =>
                {
                    var r = new Run {Text = str};
                    var para = new Paragraph
                    {
                        LineHeight = 1,
                        Margin = new Thickness(0),
                        Foreground = color,
                        FontWeight = FontWeights.Normal
                    };
                    para.Inlines.Add(r);
                    var cell = new TableCell(para);
                    tabelRow.Cells.Add(cell);
                });
                rowGroup.Rows.Add(tabelRow);
            });
            RichTextBox.Document.Blocks.Add(tableBlock);
        }

        private void addTabel(DataTable table)
        {
            var tableBlock = new Table();

            var rowGroup = new TableRowGroup();
            tableBlock.RowGroups.Add(rowGroup);

            var columnCount = table.Columns.Count;


            var tabelHeader = new TableRow();
            var tableColumn = Enumerable.Range(0, columnCount).Select(a => table.Columns[a].ColumnName);
            tableColumn.ToList().ForEach(str =>
            {
                var r = new Run {Text = str};
                var para = new Paragraph
                {
                    LineHeight = 1,
                    Margin = new Thickness(0),
                    Foreground = new SolidColorBrush(Colors.Black),
                    FontWeight = FontWeights.Normal
                };
                para.Inlines.Add(r);
                var cell = new TableCell(para);
                tabelHeader.Cells.Add(cell);
            });
            rowGroup.Rows.Add(tabelHeader);


            foreach (var row in table.Rows)
            {
                var tabelRow = new TableRow();
                var datarow = (DataRow) row;
                Enumerable.Range(0, columnCount)
                    .Select(a => datarow[a].ToString())
                    .ToList()
                    .ForEach(str =>
                    {
                        var r = new Run {Text = str};
                        var para = new Paragraph
                        {
                            LineHeight = 1,
                            Margin = new Thickness(0),
                            Foreground = new SolidColorBrush(Colors.DarkBlue),
                            FontWeight = FontWeights.Normal
                        };
                        para.Inlines.Add(r);
                        var cell = new TableCell(para);
                        tabelRow.Cells.Add(cell);
                    });
                rowGroup.Rows.Add(tabelRow);
            }

            RichTextBox.Document.Blocks.Add(tableBlock);
        }

        #endregion
    }
}