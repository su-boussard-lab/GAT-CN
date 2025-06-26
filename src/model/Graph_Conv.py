
import torch
from torch import nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.utils import expand_as_pair, check_eq_shape


class SAGEConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 feat_drop=0.0,
                 bias=True,
                 norm=None,
                 activation=F.relu,
                kernel_out = 50, 
                kernel_sizes = [1,1,1]):
        super(SAGEConv, self).__init__()
        
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        self.kernel_sizes = kernel_sizes
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
            
        if aggregator_type == 'bilstm':
            self.bilstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True, bidirectional=True) #
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)
            
        if aggregator_type != 'gcn':
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias) 

        if aggregator_type == 'cnn':
            self.convs = nn.ModuleList([nn.Conv2d(1, kernel_out, kernel_size=(window, self._in_src_feats)) for window in self.kernel_sizes])
            self.fc_convs = nn.Linear(len(self.kernel_sizes)*kernel_out, self._in_src_feats)

        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False) 
        if bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()
        self.fc_rst = nn.Linear(self._in_src_feats*2, self._in_src_feats)

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
            
        if self._aggre_type == 'bilstm':
            self.bilstm.reset_parameters() 
            
        if self._aggre_type == 'lstm':
            self.lstm.reset_parameters() 
            
        if self._aggre_type == 'cnn':
            for conv in self.convs:
                nn.init.xavier_uniform_(conv.weight, gain=gain)            
            nn.init.xavier_uniform_(self.fc_convs.weight, gain=gain)
            
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _bilstm_reducer(self, nodes):
        """bi-LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox['m'] # (B, L, D)
        batch_size = m.shape[0]
        h = (m.new_zeros((2, batch_size, self._in_src_feats)),
             m.new_zeros((2, batch_size, self._in_src_feats)))
        
        _, (rst, _) = self.bilstm(m, h)
        rst = torch.cat([rst[0, :, :], rst[1, :, :]], dim=1).unsqueeze(0)
 
        rst = self.fc_rst(rst)
        return {'neigh': rst.squeeze(0)}
    
    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox['m'] # (B, L, D)
        batch_size = m.shape[0]
        h = (m.new_zeros((1, batch_size, self._in_src_feats)),
             m.new_zeros((1, batch_size, self._in_src_feats)))
        _, (rst, _) = self.lstm(m, h)
        
        return {'neigh': rst.squeeze(0)}

    
    def _cnn_reducer(self, nodes):
        
        m = nodes.mailbox['m']
        batch_size = m.shape[0] 

        x = [conv(m.unsqueeze(1)).squeeze(3) for conv in self.convs] 
        
        output = [F.max_pool1d(x_i, (x_i.size(2))).squeeze(2) for x_i in x]
        
        output = torch.cat(output, dim = 1)
        y = self.fc_convs(output) 

        return {'neigh': y}  
        
    
    def _depth_reducer(self, nodes):

        m = nodes.mailbox['m'] 
        batch_size = m.shape[0]

        depthwise = nn.Conv2d(batch_size, batch_size, kernel_size=3, padding=1, groups=len(inputs))
        pointwise = nn.Conv2d(batch_size, 1, kernel_size=1)

        out = depthwise(torch.stack(inputs, dim=0).unsqueeze(0))
        out = pointwise(out)

        return out.squeeze(0).squeeze(0)
    
    def _compatibility_check(self):
        """Address the backward compatibility issue brought by #2747"""
        if not hasattr(self, 'bias'):
            print("You are loading a GraphSAGE model trained from a old version of DGL, "
                        "DGL automatically convert it to be compatible with latest version.")
            bias = self.fc_neigh.bias
            self.fc_neigh.bias = None
            if hasattr(self, 'fc_self'):
                if bias is not None:
                    bias = bias + self.fc_self.bias
                    self.fc_self.bias = None
            self.bias = bias

    def forward(self, graph, feat,edge_weight=None):
        r"""Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        self._compatibility_check()
        with graph.local_scope():
            
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            msg_fn = fn.copy_u('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                msg_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            h_self = feat_dst # h_v'(k-1)

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats

            # Message Passing
            if self._aggre_type == 'mean':
                graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
                graph.update_all(msg_fn, fn.mean('m', 'neigh'))
                h_neigh = graph.dstdata['neigh']
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
            elif self._aggre_type == 'gcn':
                check_eq_shape(feat)
                graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
                if isinstance(feat, tuple):
                    graph.dstdata['h'] = self.fc_neigh(feat_dst) if lin_before_mp else feat_dst
                else:
                    if graph.is_block:
                        graph.dstdata['h'] = graph.srcdata['h'][:graph.num_dst_nodes()]
                    else:
                        graph.dstdata['h'] = graph.srcdata['h']
                graph.update_all(msg_fn, fn.sum('m', 'neigh'))
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
            elif self._aggre_type == 'pool':
                graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
                graph.update_all(msg_fn, fn.max('m', 'neigh'))
                h_neigh = self.fc_neigh(graph.dstdata['neigh'])
            elif self._aggre_type == 'lstm':
                graph.srcdata['h'] = feat_src
                graph.update_all(msg_fn, self._lstm_reducer)
                h_neigh = self.fc_neigh(graph.dstdata['neigh'])
                
            elif self._aggre_type == 'bilstm':
                graph.srcdata['h'] = feat_src
                graph.update_all(msg_fn, self._bilstm_reducer)
                h_neigh = self.fc_neigh(graph.dstdata['neigh'])

            elif self._aggre_type == 'cnn':
                graph.srcdata['h'] = feat_src
                graph.update_all(msg_fn, self._cnn_reducer)
                h_neigh = self.fc_neigh(graph.dstdata['neigh'])     
            else:
                raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

            if self._aggre_type == 'gcn':
                rst = h_neigh
            else:
                rst = self.fc_self(h_self) + h_neigh 

            # bias term
            if self.bias is not None:
                rst = rst + self.bias

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
                
            
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst
