package it.unibo.model.network

import it.unibo.model.network.torch._
import me.shadaj.scalapy.py

object DQNGNN {
  def apply(input: Int, hidden: Int, output: Int): py.Dynamic = {
    val gcn = geometric.nn.GCN(input, hidden, 1, out_channels = output)
    /*nn.Sequential(
      nn.Linear(input, hidden),
      nn.ReLU(),
      nn.Linear(hidden, hidden),
      nn.ReLU(),
      nn.Linear(hidden, output)
    )*/
    gcn
  }
}
