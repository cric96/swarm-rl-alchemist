package it.unibo.model.network

import it.unibo.model.network.torch._
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.Any.from
import me.shadaj.scalapy.py.SeqConverters

object DQNGNN {
  def apply(input: Int, hidden: Int, output: Int): py.Dynamic = {
    val module = geometric.nn.Sequential(
      "x, edge_index",
      Seq(
        (geometric.nn.GCN(input, hidden, 1), "x, edge_index -> x".as[py.Any]).as[py.Any],
        torch.nn.ReLU(inplace = true),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(inplace = true),
        torch.nn.Linear(hidden, output)
      ).toPythonCopy
    )
    // val gcn = geometric.nn.EdgeCNN(input, hidden, 1, out_channels = output)
    /*nn.Sequential(
      nn.Linear(input, hidden),
      nn.ReLU(),
      nn.Linear(hidden, hidden),
      nn.ReLU(),
      nn.Linear(hidden, output)
    )*/
    module
  }
}
