import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import numpy as np
from torch.nn import Parameter
import pdb
from networks import deeplab_xception, gcn, deeplab_xception_synBN, deeplab_xception_insBN
pascal_adj = np.array([[1,0,0,0,0,0,0],
                       [0,1,1,0,0,0,0],
                       [0,1,1,1,0,1,0],
                       [0,0,1,1,1,0,0],
                       [0,0,0,1,1,0,0],
                       [0,0,1,0,0,1,1],
                       [0,0,0,0,0,1,1]])

'''
let me make a gcn verson control
1)
'''


class DeepLabv3_plus_gcn(deeplab_xception.DeepLabv3_plus):
    def __init__(self,nInputChannels=3, n_classes=7, os=16,input_channels=256,hidden_layers=512,out_channels=256,adj1=None,adj2=None):
        super(DeepLabv3_plus_gcn, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes, os=os)
        self.featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,hidden_layers=hidden_layers,
                                                           nodes=n_classes)
        self.graph_conv1 = gcn.GraphConvolution(hidden_layers,hidden_layers)
        self.gcn_encode = gcn.Graph_trans(in_features=hidden_layers,out_features=hidden_layers,
                                          begin_nodes=n_classes,end_nodes=2)
        self.graph_conv2 = gcn.GraphConvolution(hidden_layers,hidden_layers)
        self.gcn_decode = gcn.Graph_trans(hidden_layers,hidden_layers,begin_nodes=2,end_nodes=7)
        self.graph_2_fea = gcn.Graph_to_Featuremaps(input_channels=input_channels,output_channels=out_channels,
                                                    hidden_layers=hidden_layers,nodes=n_classes
                                                    )
        self.semantic = nn.Conv2d(256*2, n_classes, kernel_size=1, stride=1)
        # self.adj1 = adj1
        # self.adj2 = adj2

    def forward(self, input,adj1,adj2):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, scale_factor=4, mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)
        ### add graph
        # print('x size',x.size(),adj1.size())
        graph = self.featuremap_2_graph(x)
        # print(graph.size())
        graph = self.graph_conv1.forward(graph,adj=adj1,relu=True)
        # print(graph.size())
        graph = self.gcn_encode.forward(graph,relu=True)
        graph = self.graph_conv2.forward(graph,adj=adj2,relu=True)
        graph = self.gcn_decode.forward(graph,relu=True)
        graph = self.graph_2_fea.forward(graph,x)

        x = torch.cat((x,graph),dim=1)
        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def get_graph_parameter(self):
        l = []
        for name,k in self.named_parameters():
            if 'graph' in name:
                l.append(k)
        return l

    def get_semantic_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'semantic' in name:
                l.append(k)
        return l

    def get_baseline_parameter(self):
        l = []
        for name,k in self.named_parameters():
            if ('graph' not in name):
                l.append(k)
        return l

    def get_graph_branch(self,selected_name = ''):
        l = []
        for name,k in self.named_parameters():
            if selected_name in name:
                l.append(k)
        return l

class DeepLabv3_plus_symgcn(deeplab_xception.DeepLabv3_plus):
    def __init__(self,nInputChannels=3, n_classes=7, os=16,input_channels=256,hidden_layers=128,out_channels=256,adj1=None,adj2=None):
        super(DeepLabv3_plus_symgcn, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes, os=os)
        self.featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,hidden_layers=hidden_layers,
                                                           nodes=n_classes)
        self.graph_conv1 = gcn.GraphConvolution(hidden_layers,hidden_layers)
        # self.gcn_encode = gcn.Graph_trans(in_features=hidden_layers,out_features=hidden_layers,
        #                                   begin_nodes=n_classes,end_nodes=2)
        # self.graph_conv2 = gcn.GraphConvolution(hidden_layers,hidden_layers)
        # self.gcn_decode = gcn.Graph_trans(hidden_layers,hidden_layers,begin_nodes=2,end_nodes=7)
        self.graph_2_fea = gcn.Graph_to_Featuremaps_mhp_s(input_channels=input_channels,output_channels=out_channels,
                                                    hidden_layers=hidden_layers,nodes=n_classes
                                                    )
        self.semantic = nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
        # self.adj1 = adj1
        # self.adj2 = adj2\

    def forward(self, input,adj1,adj2):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, scale_factor=4, mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)
        ### add graph
        # print('x size',x.size(),adj1.size())
        graph = self.featuremap_2_graph(x)
        # pdb.set_trace()
        # print(graph.size())
        graph = self.graph_conv1.forward(graph,adj=adj1,relu=True)
        # pdb.set_trace()
        # print(graph.size())
        # graph = self.gcn_encode.forward(graph,relu=True)
        # graph = self.graph_conv2.forward(graph,adj=adj2,relu=True)
        # graph = self.gcn_decode.forward(graph,relu=True)
        graph = self.graph_2_fea.forward(graph,x)
        # pdb.set_trace()
        # x = torch.cat((x,graph),dim=1)
        # x = x + graph
        ###
        x = self.semantic(x)
        print(x.size())
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def get_graph_parameter(self):
        l = []
        for name,k in self.named_parameters():
            if 'graph' in name:
                l.append(k)
        return l

    def get_semantic_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'semantic' in name:
                l.append(k)
        return l

class DeepLabv3_plus_symgcn1(deeplab_xception.DeepLabv3_plus):
    def __init__(self,nInputChannels=3, n_classes=7, os=16,input_channels=256,hidden_layers=128,out_channels=256,adj1=None,adj2=None):
        super(DeepLabv3_plus_symgcn1, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes, os=os)
        self.featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,hidden_layers=hidden_layers,
                                                           nodes=n_classes)
        self.graph_conv1 = gcn.GraphConvolution(hidden_layers,hidden_layers)
        # self.gcn_encode = gcn.Graph_trans(in_features=hidden_layers,out_features=hidden_layers,
        #                                   begin_nodes=n_classes,end_nodes=2)
        # self.graph_conv2 = gcn.GraphConvolution(hidden_layers,hidden_layers)
        # self.gcn_decode = gcn.Graph_trans(hidden_layers,hidden_layers,begin_nodes=2,end_nodes=7)
        self.graph_2_fea = gcn.Graph_to_Featuremaps(input_channels=input_channels,output_channels=out_channels,
                                                    hidden_layers=hidden_layers,nodes=n_classes
                                                    )
        self.semantic = nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
        # self.adj1 = adj1
        # self.adj2 = adj2\

    def forward(self, input,adj1,adj2):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, scale_factor=4, mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)
        ### add graph
        # print('x size',x.size(),adj1.size())
        graph = self.featuremap_2_graph(x)
        # pdb.set_trace()
        # print(graph.size())
        graph = self.graph_conv1.forward(graph,adj=adj1,relu=True)
        # pdb.set_trace()
        # print(graph.size())
        # graph = self.gcn_encode.forward(graph,relu=True)
        # graph = self.graph_conv2.forward(graph,adj=adj2,relu=True)
        # graph = self.gcn_decode.forward(graph,relu=True)
        graph = self.graph_2_fea.forward(graph,x)
        # pdb.set_trace()
        # x = torch.cat((x,graph),dim=1)
        # x = x + graph
        ###
        x = self.semantic(x)
        print(x.size())
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def get_graph_parameter(self):
        l = []
        for name,k in self.named_parameters():
            if 'graph' in name:
                l.append(k)
        return l

    def get_semantic_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'semantic' in name:
                l.append(k)
        return l

class DeepLabv3_plus_gcn_v2(deeplab_xception.DeepLabv3_plus):
    def __init__(self,nInputChannels=3, n_classes=7, os=16,input_channels=256,hidden_layers=512,out_channels=256,
                 adj1=None,adj2=None,hidden_nodes=2,):
        super(DeepLabv3_plus_gcn_v2, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes, os=os)
        self.featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,hidden_layers=hidden_layers,
                                                           nodes=n_classes)
        self.graph_conv1 = gcn.GraphConvolution(hidden_layers,hidden_layers)
        # v2 add graph conv skip connection
        self.graph_conv_skip = gcn.GraphConvolution(hidden_layers,hidden_layers)
        # v2 add graph conv to after decoding graph
        self.graph_conv_final = gcn.GraphConvolution(hidden_layers,hidden_layers)
        self.gcn_encode = gcn.Graph_trans(in_features=hidden_layers,out_features=hidden_layers,
                                          begin_nodes=n_classes,end_nodes=hidden_nodes)
        self.graph_conv2 = gcn.GraphConvolution(hidden_layers,hidden_layers)
        self.gcn_decode = gcn.Graph_trans(hidden_layers,hidden_layers,begin_nodes=hidden_nodes,end_nodes=n_classes)
        self.graph_2_fea = gcn.Graph_to_Featuremaps(input_channels=input_channels,output_channels=out_channels,
                                                    hidden_layers=hidden_layers,nodes=n_classes
                                                    )
        self.semantic = nn.Conv2d(256*2, n_classes, kernel_size=1, stride=1)
        # self.adj1 = adj1
        # self.adj2 = adj2

    def forward(self, input,adj1,adj2):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, scale_factor=4, mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)
        ### add graph
        # print('x size',x.size(),adj1.size())
        graph = self.featuremap_2_graph(x)
        # print(graph.size())
        graph1 = self.graph_conv1.forward(graph,adj=adj1,relu=True)
        # print(graph.size())
        graph = self.gcn_encode.forward(graph1,relu=True,adj_return=True)
        adj2_new = self.gcn_encode.get_adj_mat()
        graph = self.graph_conv2.forward(graph,adj=adj2_new,relu=True)
        graph = self.gcn_decode.forward(graph,relu=True)
        # v2  add-- skip connection &
        graph = graph + self.graph_conv_skip.forward(graph1,relu=True)
        graph = self.graph_conv_final.forward(graph,adj=adj1,relu=True)

        graph = self.graph_2_fea.forward(graph,x)

        x = torch.cat((x,graph),dim=1)
        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def get_graph_parameter(self):
        l = []
        for name,k in self.named_parameters():
            if 'graph' in name:
                l.append(k)
        return l

    def get_semantic_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'semantic' in name:
                l.append(k)
        return l

class DeepLabv3_plus_gcn_v2_s(deeplab_xception.DeepLabv3_plus):
    def __init__(self,nInputChannels=3, n_classes=7, os=16,input_channels=256,hidden_layers=512,out_channels=256,
                 adj1=None,adj2=None,hidden_nodes=2,):
        super(DeepLabv3_plus_gcn_v2_s, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes, os=os)
        self.featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,hidden_layers=hidden_layers,
                                                           nodes=n_classes)
        self.graph_conv1 = gcn.GraphConvolution(hidden_layers,hidden_layers)
        # v2 add graph conv skip connection
        self.graph_conv_skip = gcn.GraphConvolution(hidden_layers,hidden_layers)
        # v2 add graph conv to after decoding graph
        self.graph_conv_final = gcn.GraphConvolution(hidden_layers,hidden_layers)
        self.gcn_encode = gcn.Graph_trans(in_features=hidden_layers,out_features=hidden_layers,
                                          begin_nodes=n_classes,end_nodes=hidden_nodes)
        self.graph_conv2 = gcn.GraphConvolution(hidden_layers,hidden_layers)
        self.gcn_decode = gcn.Graph_trans(hidden_layers,hidden_layers,begin_nodes=hidden_nodes,end_nodes=n_classes)
        self.graph_2_fea = gcn.Graph_to_Featuremaps(input_channels=input_channels,output_channels=out_channels,
                                                    hidden_layers=hidden_layers,nodes=n_classes
                                                    )
        self.semantic = nn.Conv2d(256*2, n_classes, kernel_size=1, stride=1)

    def forward(self, input,adj1,adj2):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        # x channel is 1280
        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, scale_factor=4, mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)

        x = self.decoder(x)

        ### add graph
        # print('x size',x.size(),adj1.size())
        graph = self.featuremap_2_graph(x)
        # print(graph.size())
        graph1 = self.graph_conv1.forward(graph, adj=adj1, relu=True)
        # print(graph.size())
        graph = self.gcn_encode.forward(graph1, relu=True, adj_return=True)
        adj2_new = self.gcn_encode.get_adj_mat()
        graph = self.graph_conv2.forward(graph, adj=adj2_new, relu=True)
        graph = self.gcn_decode.forward(graph, relu=True)
        # v2  add-- skip connection &
        graph = graph + self.graph_conv_skip.forward(graph1, relu=True)
        graph = self.graph_conv_final.forward(graph, adj=adj1, relu=True)

        graph = self.graph_2_fea.forward(graph, x)

        x = torch.cat((x,graph),dim=1)
        # end graph
        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

class DeepLabv3_plus_gcn_v2_a(deeplab_xception.DeepLabv3_plus):
    def __init__(self,nInputChannels=3, n_classes=7, os=16,input_channels=304,hidden_layers=512,out_channels=304,
                 adj1=None,adj2=None,hidden_nodes=2,):
        super(DeepLabv3_plus_gcn_v2_a, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes, os=os)
        self.featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,hidden_layers=hidden_layers,
                                                           nodes=n_classes)
        self.graph_conv1 = gcn.GraphConvolution(hidden_layers,hidden_layers)
        # v2 add graph conv skip connection
        self.graph_conv_skip = gcn.GraphConvolution(hidden_layers,hidden_layers)
        # v2 add graph conv to after decoding graph
        self.graph_conv_final = gcn.GraphConvolution(hidden_layers,hidden_layers)
        self.gcn_encode = gcn.Graph_trans(in_features=hidden_layers,out_features=hidden_layers,
                                          begin_nodes=n_classes,end_nodes=hidden_nodes)
        self.graph_conv2 = gcn.GraphConvolution(hidden_layers,hidden_layers)
        self.gcn_decode = gcn.Graph_trans(hidden_layers,hidden_layers,begin_nodes=hidden_nodes,end_nodes=n_classes)
        self.graph_2_fea = gcn.Graph_to_Featuremaps(input_channels=input_channels,output_channels=out_channels,
                                                    hidden_layers=hidden_layers,nodes=n_classes
                                                    )
        self.semantic = nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
        # self.adj1 = adj1
        # self.adj2 = adj2

    def forward(self, input,adj1,adj2):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        # x channel is 1280
        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, scale_factor=4, mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)

        ### add graph
        # print('x size',x.size(),adj1.size())
        graph = self.featuremap_2_graph(x)
        # print(graph.size())
        graph1 = self.graph_conv1.forward(graph, adj=adj1, relu=True)
        # print(graph.size())
        graph = self.gcn_encode.forward(graph1, relu=True, adj_return=True)
        adj2_new = self.gcn_encode.get_adj_mat()
        graph = self.graph_conv2.forward(graph, adj=adj2_new, relu=True)
        graph = self.gcn_decode.forward(graph, relu=True)
        # v2  add-- skip connection &
        graph = graph + self.graph_conv_skip.forward(graph1, relu=True)
        graph = self.graph_conv_final.forward(graph, adj=adj1, relu=True)

        graph = self.graph_2_fea.forward(graph, x)

        x = x + graph
        # end graph
        ###

        x = self.decoder(x)

        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def get_graph_parameter(self):
        l = []
        for name,k in self.named_parameters():
            if 'graph' in name:
                l.append(k)
        return l

    def get_semantic_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'semantic' in name:
                l.append(k)
        return l

class DeepLabv3_plus_gcn_v2_cihp2pascal(deeplab_xception.DeepLabv3_plus):
    def __init__(self,nInputChannels=3, n_classes=7, os=16,input_channels=256,hidden_layers=512,out_channels=256,adj1=None,adj2=None,hidden_nodes=4):
        super(DeepLabv3_plus_gcn_v2_cihp2pascal, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes, os=os)
        # pascal
        self.pascal_featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels, hidden_layers=hidden_layers,
                                                           nodes=n_classes)
        self.pascal_graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.pascal_graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.pascal_graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        # encode & decode
        self.pascal_gcn_encode = gcn.Graph_trans(in_features=hidden_layers, out_features=hidden_layers,
                                          begin_nodes=n_classes, end_nodes=hidden_nodes)

        self.pascal_gcn_decode = gcn.Graph_trans(hidden_layers, hidden_layers, begin_nodes=hidden_nodes, end_nodes=n_classes)
        # graph to feature map
        self.pascal_graph_2_fea = gcn.Graph_to_Featuremaps(input_channels=input_channels, output_channels=out_channels,
                                                    hidden_layers=hidden_layers, nodes=n_classes
                                                    )
        # v2 add graph conv skip connection
        self.pascal_graph_conv_skip = gcn.GraphConvolution(hidden_layers, hidden_layers)
        # pre feature skip conv
        self.pascal_skip_conv = nn.Sequential(*[nn.Conv2d(input_channels, input_channels, kernel_size=1),
                                         nn.ReLU(True)])
        self.semantic = nn.Conv2d(256, n_classes, kernel_size=1, stride=1)


        # cihp
        self.featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels, hidden_layers=hidden_layers,
                                                           nodes=20)
        self.graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)

        # encode & decode
        self.gcn_encode = gcn.Graph_trans(in_features=hidden_layers, out_features=hidden_layers,
                                          begin_nodes=20, end_nodes=hidden_nodes)



        # connection fc
        self.graph_connection_fc = nn.Linear(hidden_layers,hidden_layers)
        #

        self.semantic = nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
        # self.adj1 = adj1
        # self.adj2 = adj2

    def forward(self, input,adj1,adj2,adj3):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, scale_factor=4, mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)
        ### add graph
        graph = self.pascal_featuremap_2_graph(x)
        # print(graph.size())
        graph = self.pascal_graph_conv1.forward(graph, adj=adj1, relu=True)
        graph = self.pascal_graph_conv1.forward(graph, adj=adj1, relu=True)
        graph1 = self.pascal_graph_conv1.forward(graph, adj=adj1, relu=True)
        # print(graph.size())
        graph = self.pascal_gcn_encode.forward(graph1, relu=True, adj_return=True)

        ## cihp begin
        # print('x size',x.size(),adj1.size())
        graph_cihp = self.featuremap_2_graph(x)
        # print(graph.size())
        graph_cihp = self.graph_conv1.forward(graph_cihp, adj=adj3, relu=True)
        graph_cihp = self.graph_conv1.forward(graph_cihp, adj=adj3, relu=True)
        graph1_cihp = self.graph_conv1.forward(graph_cihp, adj=adj3, relu=True)
        graph_cihp = self.gcn_encode.forward(graph1_cihp, relu=True, adj_return=True)
        # new_adj2 = self.gcn_encode.get_adj_mat()
        ### cihp end

        combine_graph = self.node_feature_combine(graph_cihp,graph)


        new_adj2 = self.pascal_gcn_encode.get_adj_mat()
        # graph conv 2
        graph = self.pascal_graph_conv2.forward(graph, adj=new_adj2, relu=True)
        graph = self.pascal_graph_conv2.forward(graph, adj=new_adj2, relu=True)
        decode_adj_mat = self.pascal_gcn_encode.get_encode_adj()
        graph = self.pascal_gcn_decode.forward(graph, relu=True, adj=decode_adj_mat.transpose(0, 1))
        graph = self.pascal_graph_conv_skip.forward(graph1, relu=True) + graph
        # graph conv 3
        graph = self.pascal_graph_conv3.forward(graph, adj=adj1, relu=True)
        graph = self.pascal_graph_conv3.forward(graph, adj=adj1, relu=True)
        graph = self.pascal_graph_conv3.forward(graph, adj=adj1, relu=True)

        graph = self.pascal_graph_2_fea.forward(graph, x)
        x = self.pascal_skip_conv(x)
        x = x + graph



        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def get_graph_parameter(self):
        l = []
        for name,k in self.named_parameters():
            if 'graph' in name:
                l.append(k)
        return l

    def get_semantic_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'semantic' in name:
                l.append(k)
        return l

    def get_baseline_parameter(self):
        l = []
        for name,k in self.named_parameters():
            if ('graph' not in name):
                l.append(k)
        return l

    def get_graph_branch(self,selected_name = ''):
        l = []
        for name,k in self.named_parameters():
            if selected_name in name:
                l.append(k)
        return l

    def node_feature_combine(self,input1,input2):
        input11 = F.normalize(input1,p=2,dim=-1)
        input22 = F.normalize(input2,p=2,dim=-1)
        # input1 -> input2
        relation = torch.matmul(input11,input22.transpose(-1,-2))
        output = torch.matmul(relation.transpose(-1,-2),input1)
        return output+input2

class DeepLabv3_plus_gcn_v2_cihp2pascal_kp(DeepLabv3_plus_gcn_v2_cihp2pascal):
    def __init__(self,nInputChannels=3, n_classes=7, os=16,input_channels=256,hidden_layers=128,out_channels=256,adj1=None,adj2=None,hidden_nodes=4):
        super(DeepLabv3_plus_gcn_v2_cihp2pascal_kp, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes, os=16,input_channels=input_channels,hidden_layers=hidden_layers,out_channels=out_channels,hidden_nodes=hidden_nodes)

    def forward(self, input,adj1,adj_transfer,adj3):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)
        ### add graph

        ## cihp begin
        # print('x size',x.size(),adj1.size())
        graph_cihp = self.featuremap_2_graph(x)
        # print(graph.size())
        graph_cihp = self.graph_conv1.forward(graph_cihp, adj=adj3, relu=True)
        graph_cihp = self.graph_conv1.forward(graph_cihp, adj=adj3, relu=True)
        graph1_cihp = self.graph_conv1.forward(graph_cihp, adj=adj3, relu=True)
        graph_cihp = torch.matmul(adj_transfer, graph1_cihp)
        # new_adj2 = self.gcn_encode.get_adj_mat()
        ### cihp end

        graph = self.pascal_featuremap_2_graph(x)
        # print(graph.size())
        graph = graph + graph_cihp
        graph = self.pascal_graph_conv1.forward(graph, adj=adj1, relu=True)
        graph = self.pascal_graph_conv1.forward(graph, adj=adj1, relu=True)
        graph1 = self.pascal_graph_conv1.forward(graph, adj=adj1, relu=True)
        # print(graph.size())
        graph = self.pascal_gcn_encode.forward(graph1, relu=True, adj_return=True)



        new_adj2 = self.pascal_gcn_encode.get_adj_mat()
        # graph conv 2
        graph = self.pascal_graph_conv2.forward(graph, adj=new_adj2, relu=True)
        graph = self.pascal_graph_conv2.forward(graph, adj=new_adj2, relu=True)
        decode_adj_mat = self.pascal_gcn_encode.get_encode_adj()
        graph = self.pascal_gcn_decode.forward(graph, relu=True, adj=decode_adj_mat.transpose(0, 1))
        graph = self.pascal_graph_conv_skip.forward(graph1, relu=True) + graph
        # graph conv 3
        graph = self.pascal_graph_conv3.forward(graph, adj=adj1, relu=True)
        graph = self.pascal_graph_conv3.forward(graph, adj=adj1, relu=True)
        graph = self.pascal_graph_conv3.forward(graph, adj=adj1, relu=True)

        graph = self.pascal_graph_2_fea.forward(graph, x)
        x = self.pascal_skip_conv(x)
        x = x + graph

        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

######################################################################################


class DeepLabv3_plus_symgcn_copy(deeplab_xception.DeepLabv3_plus):
    def __init__(self,nInputChannels=3, n_classes=7, os=16,input_channels=256,hidden_layers=128,out_channels=256,adj1=None,adj2=None):
        super(DeepLabv3_plus_symgcn_copy, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes, os=os)
        self.featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,hidden_layers=hidden_layers,
                                                           nodes=n_classes)
        self.graph_conv1 = gcn.GraphConvolution(hidden_layers,hidden_layers)
        self.graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)

        self.graph_2_fea = gcn.Graph_to_Featuremaps(input_channels=input_channels,output_channels=out_channels,
                                                    hidden_layers=hidden_layers,nodes=n_classes
                                                    )
        self.skip_conv = nn.Sequential(*[nn.Conv2d(input_channels,input_channels,kernel_size=1),
                                         nn.ReLU(True)])
        self.semantic = nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
        # self.adj1 = adj1
        # self.adj2 = adj2\

    def forward(self, input, adj1=None, adj2=None):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)
        ### add graph
        # print('x size',x.size(),adj1.size())
        graph = self.featuremap_2_graph(x)
        # print(graph.size())
        graph = self.graph_conv1.forward(graph, adj=adj1, relu=True)
        graph = self.graph_conv2.forward(graph, adj=adj1, relu=True)
        graph = self.graph_conv3.forward(graph, adj=adj1, relu=True)
        # print(graph.size(),x.size())
        # graph = self.gcn_encode.forward(graph,relu=True)
        # graph = self.graph_conv2.forward(graph,adj=adj2,relu=True)
        # graph = self.gcn_decode.forward(graph,relu=True)
        graph = self.graph_2_fea.forward(graph,x)
        x = self.skip_conv(x)
        x = x + graph
        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def get_graph_parameter(self):
        l = []
        for name,k in self.named_parameters():
            if 'graph' in name:
                l.append(k)
        return l

    def get_semantic_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'semantic' in name:
                l.append(k)
        return l

class DeepLabv3_plus_symgcn_copy_synbn(deeplab_xception_synBN.DeepLabv3_plus):
    def __init__(self,nInputChannels=3, n_classes=7, os=16,input_channels=256,hidden_layers=128,out_channels=256,adj1=None,adj2=None):
        super(DeepLabv3_plus_symgcn_copy_synbn, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes, os=os)
        self.featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,hidden_layers=hidden_layers,
                                                           nodes=n_classes)
        self.graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)

        self.graph_2_fea = gcn.Graph_to_Featuremaps(input_channels=input_channels,output_channels=out_channels,
                                                    hidden_layers=hidden_layers,nodes=n_classes
                                                    )
        self.skip_conv = nn.Sequential(*[nn.Conv2d(input_channels,input_channels,kernel_size=1),
                                         nn.ReLU(True)])
        self.semantic = nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
        # self.adj1 = adj1
        # self.adj2 = adj2\

    def forward(self, input, adj1=None, adj2=None):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)
        ### add graph
        # print('x size',x.size(),adj1.size())
        graph = self.featuremap_2_graph(x)
        # print(graph.size())
        graph = self.graph_conv1.forward(graph, adj=adj1, relu=True)
        graph = self.graph_conv2.forward(graph, adj=adj1, relu=True)
        graph = self.graph_conv3.forward(graph, adj=adj1, relu=True)
        # print(graph.size(),x.size())
        # graph = self.gcn_encode.forward(graph,relu=True)
        # graph = self.graph_conv2.forward(graph,adj=adj2,relu=True)
        # graph = self.gcn_decode.forward(graph,relu=True)
        graph = self.graph_2_fea.forward(graph,x)
        x = self.skip_conv(x)
        x = x + graph
        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def get_graph_parameter(self):
        l = []
        for name,k in self.named_parameters():
            if 'graph' in name:
                l.append(k)
        return l

    def get_semantic_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'semantic' in name:
                l.append(k)
        return l

class DeepLabv3_plus_symgcn_copy_20(deeplab_xception.DeepLabv3_plus):
    def __init__(self,nInputChannels=3, n_classes=7, os=16,input_channels=256,hidden_layers=128,out_channels=256,adj1=None,adj2=None):
        super(DeepLabv3_plus_symgcn_copy_20, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes, os=os)
        self.featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,hidden_layers=hidden_layers,
                                                           nodes=20)
        self.graph_conv1 = gcn.GraphConvolution(hidden_layers,hidden_layers)
        self.graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)

        self.graph_2_fea = gcn.Graph_to_Featuremaps(input_channels=input_channels,output_channels=out_channels,
                                                    hidden_layers=hidden_layers,nodes=20
                                                    )
        self.skip_conv = nn.Sequential(*[nn.Conv2d(input_channels,input_channels,kernel_size=1),
                                         nn.ReLU(True)])
        self.semantic = nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
        # self.adj1 = adj1
        # self.adj2 = adj2\

    def forward(self, input,adj1,adj2):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)
        ### add graph
        # print('x size',x.size(),adj1.size())
        graph = self.featuremap_2_graph(x)
        # print(graph.size())
        graph = self.graph_conv1.forward(graph, adj=adj1, relu=True)
        graph = self.graph_conv2.forward(graph, adj=adj1, relu=True)
        graph = self.graph_conv3.forward(graph, adj=adj1, relu=True)
        # print(graph.size(),x.size())
        # graph = self.gcn_encode.forward(graph,relu=True)
        # graph = self.graph_conv2.forward(graph,adj=adj2,relu=True)
        # graph = self.gcn_decode.forward(graph,relu=True)
        graph = self.graph_2_fea.forward(graph,x)
        x = self.skip_conv(x)
        x = x + graph
        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def get_graph_parameter(self):
        l = []
        for name,k in self.named_parameters():
            if 'graph' in name:
                l.append(k)
        return l

    def get_semantic_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'semantic' in name:
                l.append(k)
        return l

class DeepLabv3_plus_symgcn_copy_beforeaspp(deeplab_xception.DeepLabv3_plus):
    def __init__(self,nInputChannels=3, n_classes=7, os=16,input_channels=2048,hidden_layers=512,out_channels=2048,adj1=None,adj2=None):
        super(DeepLabv3_plus_symgcn_copy_beforeaspp, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes, os=os)
        self.featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,hidden_layers=hidden_layers,
                                                           nodes=n_classes)
        self.graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)

        self.graph_2_fea = gcn.Graph_to_Featuremaps(input_channels=input_channels,output_channels=out_channels,
                                                    hidden_layers=hidden_layers,nodes=n_classes
                                                    )
        self.skip_conv = nn.Sequential(*[nn.Conv2d(input_channels,out_channels,kernel_size=1),
                                         nn.ReLU(True)])
        self.semantic = nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
        # self.adj1 = adj1
        # self.adj2 = adj2\

    def forward(self, input,adj1,adj2):
        x, low_level_features = self.xception_features(input)

        ### add graph
        # print('x size',x.size(),adj1.size())
        graph = self.featuremap_2_graph(x)
        # print(graph.size())
        graph = self.graph_conv1.forward(graph, adj=adj1, relu=True)
        graph = self.graph_conv2.forward(graph, adj=adj1, relu=True)
        graph = self.graph_conv3.forward(graph, adj=adj1, relu=True)
        # print(graph.size())
        # graph = self.gcn_encode.forward(graph,relu=True)
        # graph = self.graph_conv2.forward(graph,adj=adj2,relu=True)
        # graph = self.gcn_decode.forward(graph,relu=True)
        graph = self.graph_2_fea.forward(graph, x)
        x = self.skip_conv(x)
        x = x + graph
        ###

        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, scale_factor=4, mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)

        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def get_graph_parameter(self):
        l = []
        for name,k in self.named_parameters():
            if 'graph' in name:
                l.append(k)
        return l

    def get_semantic_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'semantic' in name:
                l.append(k)
        return l

class DeepLabv3_plus_symgcn_copy_onegraph(deeplab_xception.DeepLabv3_plus):
    def __init__(self,nInputChannels=3, n_classes=7, os=16,input_channels=256,hidden_layers=128,out_channels=256,adj1=None,adj2=None):
        super(DeepLabv3_plus_symgcn_copy_onegraph, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes, os=os)
        self.featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,hidden_layers=hidden_layers,
                                                           nodes=n_classes)
        self.graph_conv1 = gcn.GraphConvolution(hidden_layers,hidden_layers)
        # self.graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        # self.graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)

        self.graph_2_fea = gcn.Graph_to_Featuremaps(input_channels=input_channels,output_channels=out_channels,
                                                    hidden_layers=hidden_layers,nodes=n_classes
                                                    )
        self.skip_conv = nn.Sequential(*[nn.Conv2d(input_channels,input_channels,kernel_size=1),
                                         nn.ReLU(True)])
        self.semantic = nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
        # self.adj1 = adj1
        # self.adj2 = adj2\

    def forward(self, input,adj1,adj2):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, scale_factor=4, mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)
        ### add graph
        # print('x size',x.size(),adj1.size())
        graph = self.featuremap_2_graph(x)
        # print(graph.size())
        graph = self.graph_conv1.forward(graph,adj=adj1,relu=True)
        graph = self.graph_conv1.forward(graph, adj=adj1, relu=True)
        graph = self.graph_conv1.forward(graph, adj=adj1, relu=True)
        # print(graph.size())
        # graph = self.gcn_encode.forward(graph,relu=True)
        # graph = self.graph_conv2.forward(graph,adj=adj2,relu=True)
        # graph = self.gcn_decode.forward(graph,relu=True)
        graph = self.graph_2_fea.forward(graph,x)
        x = self.skip_conv(x)
        x = x + graph
        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def get_graph_parameter(self):
        l = []
        for name,k in self.named_parameters():
            if 'graph' in name:
                l.append(k)
        return l

    def get_semantic_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'semantic' in name:
                l.append(k)
        return l

class DeepLabv3_plus_gcn_copy_onegraph(deeplab_xception.DeepLabv3_plus):
    def __init__(self,nInputChannels=3, n_classes=7, os=16,input_channels=256,hidden_layers=128,out_channels=256,adj1=None,adj2=None,hidden_nodes=4):
        super(DeepLabv3_plus_gcn_copy_onegraph, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes, os=os)
        self.featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,hidden_layers=hidden_layers,
                                                           nodes=n_classes)
        self.graph_conv1 = gcn.GraphConvolution(hidden_layers,hidden_layers)
        self.graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        # encode & decode
        self.gcn_encode = gcn.Graph_trans(in_features=hidden_layers, out_features=hidden_layers,
                                          begin_nodes=n_classes, end_nodes=hidden_nodes)

        self.gcn_decode = gcn.Graph_trans(hidden_layers, hidden_layers, begin_nodes=hidden_nodes, end_nodes=n_classes)
        # graph to feature map
        self.graph_2_fea = gcn.Graph_to_Featuremaps(input_channels=input_channels,output_channels=out_channels,
                                                    hidden_layers=hidden_layers,nodes=n_classes
                                                    )
        # v2 add graph conv skip connection
        self.graph_conv_skip = gcn.GraphConvolution(hidden_layers, hidden_layers)
        # pre feature skip conv
        self.skip_conv = nn.Sequential(*[nn.Conv2d(input_channels,input_channels,kernel_size=1),
                                         nn.ReLU(True)])
        self.semantic = nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
        # self.adj1 = adj1
        # self.adj2 = adj2\

    def forward(self, input,adj1,adj2):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)
        ### add graph
        # print('x size',x.size(),adj1.size())
        graph = self.featuremap_2_graph(x)
        # print(graph.size())
        graph = self.graph_conv1.forward(graph,adj=adj1,relu=True)
        graph = self.graph_conv1.forward(graph, adj=adj1, relu=True)
        graph1 = self.graph_conv1.forward(graph, adj=adj1, relu=True)
        # print(graph.size())
        graph = self.gcn_encode.forward(graph1,relu=True,adj_return=True)
        new_adj2 = self.gcn_encode.get_adj_mat()
        # graph conv 2
        graph = self.graph_conv2.forward(graph,adj=new_adj2,relu=True)
        graph = self.graph_conv2.forward(graph, adj=new_adj2, relu=True)
        decode_adj_mat = self.gcn_encode.get_encode_adj()
        graph = self.gcn_decode.forward(graph,relu=True,adj=decode_adj_mat.transpose(0,1))
        graph = self.graph_conv_skip.forward(graph1,relu=True) + graph
        # graph conv 3
        graph = self.graph_conv3.forward(graph,adj=adj1,relu=True)
        graph = self.graph_conv3.forward(graph, adj=adj1, relu=True)
        graph = self.graph_conv3.forward(graph, adj=adj1, relu=True)

        graph = self.graph_2_fea.forward(graph,x)
        x = self.skip_conv(x)
        x = x + graph
        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def get_graph_parameter(self):
        l = []
        for name,k in self.named_parameters():
            if 'graph' in name:
                l.append(k)
        return l

    def get_semantic_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'semantic' in name:
                l.append(k)
        return l

class DeepLabv3_plus_gcn_asy(deeplab_xception.DeepLabv3_plus):
    def __init__(self,nInputChannels=3, n_classes=7, os=16,input_channels=256,hidden_layers=128,out_channels=256,adj1=None,adj2=None,hidden_nodes=4):
        super(DeepLabv3_plus_gcn_asy, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes, os=os)
        self.featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,hidden_layers=hidden_layers,
                                                           nodes=n_classes)
        self.graph_conv1 = gcn.GraphConvolution(hidden_layers,hidden_layers)
        self.graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        # encode & decode
        self.gcn_encode = gcn.Graph_trans(in_features=hidden_layers, out_features=hidden_layers,
                                          begin_nodes=n_classes, end_nodes=hidden_nodes)

        self.gcn_decode = gcn.Graph_trans(hidden_layers, hidden_layers, begin_nodes=hidden_nodes, end_nodes=n_classes)
        # graph to feature map
        self.graph_2_fea = gcn.Graph_to_Featuremaps(input_channels=input_channels,output_channels=out_channels,
                                                    hidden_layers=hidden_layers,nodes=n_classes
                                                    )
        # v2 add graph conv skip connection
        self.graph_conv_skip = gcn.GraphConvolution(hidden_layers, hidden_layers)
        # pre feature skip conv
        self.skip_conv = nn.Sequential(*[nn.Conv2d(input_channels,input_channels,kernel_size=1),
                                         nn.ReLU(True)])
        self.graph_weight = gcn.Graph_weight_fc(hidden_layers)
        self.semantic = nn.Conv2d(256*2, n_classes, kernel_size=1, stride=1)
        # self.adj1 = adj1
        # self.adj2 = adj2\

    def forward(self, input,adj1,adj2):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, scale_factor=4, mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)
        ### add graph
        # print('x size',x.size(),adj1.size())
        graph = self.featuremap_2_graph(x)
        # print(graph.size())
        graph = self.graph_conv1.forward(graph,adj=adj1,relu=True)
        graph = self.graph_conv1.forward(graph, adj=adj1, relu=True)
        graph1 = self.graph_conv1.forward(graph, adj=adj1, relu=True)
        # print(graph.size())
        graph = self.gcn_encode.forward(graph1,relu=True,adj_return=True)
        new_adj2 = self.graph_weight.forward(graph)
        # graph conv 2
        graph = self.graph_conv2.forward(graph,adj=new_adj2,relu=True)
        new_adj2 = self.graph_weight.forward(graph)
        graph = self.graph_conv2.forward(graph, adj=new_adj2, relu=True)
        new_adj2 = self.graph_weight.forward(graph)
        graph = self.graph_conv2.forward(graph, adj=new_adj2, relu=True)
        decode_adj_mat = self.gcn_encode.get_encode_adj()
        graph = self.gcn_decode.forward(graph,relu=True,adj=decode_adj_mat.transpose(0,1))
        graph = self.graph_conv_skip.forward(graph1,relu=True) + graph
        # graph conv 3
        graph = self.graph_conv3.forward(graph,adj=adj1,relu=True)
        graph = self.graph_conv3.forward(graph, adj=adj1, relu=True)
        graph = self.graph_conv3.forward(graph, adj=adj1, relu=True)

        graph = self.graph_2_fea.forward(graph,x)
        x = self.skip_conv(x)
        x = torch.cat((x,graph),dim=1)
        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def get_graph_parameter(self):
        l = []
        for name,k in self.named_parameters():
            if 'graph' in name:
                l.append(k)
        return l

    def get_semantic_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'semantic' in name:
                l.append(k)
        return l

class DeepLabv3_plus_symgcn_copy_mulscale(deeplab_xception.DeepLabv3_plus):
    def __init__(self,nInputChannels=3, n_classes=7, os=16,input_channels=256,hidden_layers=128,out_channels=256,adj1=None,adj2=None):
        super(DeepLabv3_plus_symgcn_copy_mulscale, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes, os=os)
        self.featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,hidden_layers=hidden_layers,
                                                           nodes=n_classes)
        self.featuremap_2_graph_low = gcn.Featuremaps_to_Graph(input_channels=input_channels, hidden_layers=hidden_layers,
                                                           nodes=n_classes)
        self.graph_conv1 = gcn.GraphConvolution(hidden_layers,hidden_layers)
        self.graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)

        self.graph_2_fea = gcn.Graph_to_Featuremaps(input_channels=input_channels,output_channels=out_channels,
                                                    hidden_layers=hidden_layers,nodes=n_classes
                                                    )
        self.skip_conv = nn.Sequential(*[nn.Conv2d(input_channels,input_channels,kernel_size=1),
                                         nn.ReLU(True)])
        self.semantic = nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
        # self.adj1 = adj1
        # self.adj2 = adj2\

    def forward(self, input,adj1,adj2):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)

        ## low graph begin
        graph_low = self.featuremap_2_graph_low.forward(x)
        graph_low = self.graph_conv2.forward(graph_low, adj=adj1, relu=True)
        graph_low = self.graph_conv2.forward(graph_low, adj=adj1, relu=True)
        graph_low = self.graph_conv2.forward(graph_low, adj=adj1, relu=True)
        ## end

        # print(x.size())
        x = F.upsample(x, scale_factor=4, mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features) # input channel : 256
        low_level_features = self.feature_projection_bn1(low_level_features)  # channels: 48
        low_level_features = self.relu(low_level_features)

        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)
        ### add graph
        # print('x size',x.size(),adj1.size())
        graph = self.featuremap_2_graph(x)
        # print(graph.size())
        graph = self.graph_conv1.forward(graph,adj=adj1,relu=True)
        graph = self.graph_conv1.forward(graph, adj=adj1, relu=True)
        graph = self.graph_conv1.forward(graph, adj=adj1, relu=True)

        # graph combine
        graph = graph + graph_low
        graph = self.graph_conv3.forward(graph, adj=adj1, relu=True)
        graph = self.graph_conv3.forward(graph, adj=adj1, relu=True)

        graph = self.graph_2_fea.forward(graph,x)
        x = self.skip_conv(x)
        x = x + graph
        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def get_graph_parameter(self):
        l = []
        for name,k in self.named_parameters():
            if 'graph' in name:
                l.append(k)
        return l

    def get_semantic_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'semantic' in name:
                l.append(k)
        return l


class DeepLabv3_plus_symgcn_copy_mhp(deeplab_xception.DeepLabv3_plus):
    def __init__(self,nInputChannels=3, n_classes=7, os=16,input_channels=256,hidden_layers=128,out_channels=256,adj1=None,adj2=None):
        super(DeepLabv3_plus_symgcn_copy_mhp, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes, os=os)
        self.featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,hidden_layers=hidden_layers,
                                                           nodes=n_classes)
        self.graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)

        self.graph_2_fea = gcn.Graph_to_Featuremaps_mhp(input_channels=input_channels,output_channels=out_channels,
                                                    hidden_layers=hidden_layers,nodes=n_classes
                                                    )
        self.skip_conv = nn.Sequential(*[nn.Conv2d(input_channels,input_channels,kernel_size=1),
                                         nn.ReLU(True)])
        self.semantic = nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
        # self.adj1 = adj1
        # self.adj2 = adj2\

    def forward(self, input, adj1=None, adj2=None):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)
        ### add graph
        # print('x size',x.size(),adj1.size())
        graph = self.featuremap_2_graph(x)
        # print(graph.size())
        graph = self.graph_conv1.forward(graph, adj=adj1, relu=True)
        graph = self.graph_conv2.forward(graph, adj=adj1, relu=True)
        graph = self.graph_conv3.forward(graph, adj=adj1, relu=True)
        # print(graph.size(),x.size())
        # graph = self.gcn_encode.forward(graph,relu=True)
        # graph = self.graph_conv2.forward(graph,adj=adj2,relu=True)
        # graph = self.gcn_decode.forward(graph,relu=True)
        graph = self.graph_2_fea.forward(graph,x)
        x = self.skip_conv(x)
        x = x + graph
        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def get_graph_parameter(self):
        l = []
        for name,k in self.named_parameters():
            if 'graph' in name:
                l.append(k)
        return l

    def get_semantic_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'semantic' in name:
                l.append(k)
        return l

class DeepLabv3_plus_symgcn_copy_mhp_fixbn(deeplab_xception.DeepLabv3_plus):
    def __init__(self,nInputChannels=3, n_classes=7, os=16,input_channels=256,hidden_layers=128,out_channels=256,adj1=None,adj2=None):
        super(DeepLabv3_plus_symgcn_copy_mhp_fixbn, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes, os=os)
        self.featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,hidden_layers=hidden_layers,
                                                           nodes=n_classes)
        self.graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)

        self.graph_2_fea = gcn.Graph_to_Featuremaps_mhp(input_channels=input_channels,output_channels=out_channels,
                                                    hidden_layers=hidden_layers,nodes=n_classes
                                                    )
        self.skip_conv = nn.Sequential(*[nn.Conv2d(input_channels,input_channels,kernel_size=1),
                                         nn.ReLU(True)])
        self.semantic = nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
        # self.adj1 = adj1
        # self.adj2 = adj2

    def forward(self, input, adj1=None, adj2=None):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)
        ### add graph
        # print('x size',x.size(),adj1.size())
        graph = self.featuremap_2_graph(x)
        # print(graph.size())
        graph = self.graph_conv1.forward(graph, adj=adj1, relu=True)
        graph = self.graph_conv2.forward(graph, adj=adj1, relu=True)
        graph = self.graph_conv3.forward(graph, adj=adj1, relu=True)
        # print(graph.size(),x.size())
        # graph = self.gcn_encode.forward(graph,relu=True)
        # graph = self.graph_conv2.forward(graph,adj=adj2,relu=True)
        # graph = self.gcn_decode.forward(graph,relu=True)
        graph = self.graph_2_fea.forward(graph,x)
        x = self.skip_conv(x)
        x = x + graph
        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def get_graph_parameter(self):
        l = []
        for name,k in self.named_parameters():
            if 'graph' in name:
                l.append(k)
        return l

    def get_semantic_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'semantic' in name or 'skip_conv' in name:
                l.append(k)
        return l

    def train(self, mode=True, freeze_bn=True, freeze_bn_affine=True):
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Returns:
            Module: self
        """
        super(DeepLabv3_plus_symgcn_copy_mhp_fixbn, self).train(mode)
        if freeze_bn:
            print("Freezing Mean/Var of BatchNorm2D.")
            if freeze_bn_affine:
                print("Freezing Weight/Bias of BatchNorm2D.")
        if freeze_bn:
            for m in self.xception_features.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if freeze_bn_affine:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
            for m in self.aspp1.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if freeze_bn_affine:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
            for m in self.aspp2.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if freeze_bn_affine:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
            for m in self.aspp3.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if freeze_bn_affine:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
            for m in self.aspp4.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if freeze_bn_affine:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
            for m in self.global_avg_pool.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if freeze_bn_affine:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
            for m in self.concat_projection_bn1.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if freeze_bn_affine:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
            for m in self.feature_projection_bn1.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if freeze_bn_affine:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False


class DeepLabv3_plus_symgcn_copy_mhp_ins(deeplab_xception_insBN.DeepLabv3_plus):
    def __init__(self,nInputChannels=3, n_classes=7, os=16,input_channels=256,hidden_layers=128,out_channels=256,adj1=None,adj2=None):
        super(DeepLabv3_plus_symgcn_copy_mhp_ins, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes, os=os)
        self.featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,hidden_layers=hidden_layers,
                                                           nodes=n_classes)
        self.graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)

        self.graph_2_fea = gcn.Graph_to_Featuremaps_mhp(input_channels=input_channels,output_channels=out_channels,
                                                    hidden_layers=hidden_layers,nodes=n_classes
                                                    )
        self.skip_conv = nn.Sequential(*[nn.Conv2d(input_channels,input_channels,kernel_size=1),
                                         nn.ReLU(True)])
        self.semantic = nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
        # self.adj1 = adj1
        # self.adj2 = adj2\

    def forward(self, input, adj1=None, adj2=None):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)
        ### add graph
        # print('x size',x.size(),adj1.size())
        graph = self.featuremap_2_graph(x)
        # print(graph.size())
        graph = self.graph_conv1.forward(graph, adj=adj1, relu=True)
        graph = self.graph_conv2.forward(graph, adj=adj1, relu=True)
        graph = self.graph_conv3.forward(graph, adj=adj1, relu=True)
        # print(graph.size(),x.size())
        # graph = self.gcn_encode.forward(graph,relu=True)
        # graph = self.graph_conv2.forward(graph,adj=adj2,relu=True)
        # graph = self.gcn_decode.forward(graph,relu=True)
        graph = self.graph_2_fea.forward(graph,x)
        x = self.skip_conv(x)
        x = x + graph
        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def get_graph_parameter(self):
        l = []
        for name,k in self.named_parameters():
            if 'graph' in name:
                l.append(k)
        return l

    def get_semantic_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'semantic' in name:
                l.append(k)
        return l


#################################################### Ablation study

class DeepLabv3_plus_symgcn_copy_1graphlayer(deeplab_xception.DeepLabv3_plus):
    def __init__(self,nInputChannels=3, n_classes=7, os=16,input_channels=256,hidden_layers=128,out_channels=256,adj1=None,adj2=None):
        super(DeepLabv3_plus_symgcn_copy_1graphlayer, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes, os=os)
        self.featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,hidden_layers=hidden_layers,
                                                           nodes=n_classes)
        self.graph_conv1 = gcn.GraphConvolution(hidden_layers,hidden_layers)

        self.graph_2_fea = gcn.Graph_to_Featuremaps(input_channels=input_channels,output_channels=out_channels,
                                                    hidden_layers=hidden_layers,nodes=n_classes
                                                    )
        self.skip_conv = nn.Sequential(*[nn.Conv2d(input_channels,input_channels,kernel_size=1),
                                         nn.ReLU(True)])
        self.semantic = nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
        # self.adj1 = adj1
        # self.adj2 = adj2\

    def forward(self, input, adj1=None, adj2=None):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)
        ### add graph
        # print('x size',x.size(),adj1.size())
        graph = self.featuremap_2_graph(x)
        # print(graph.size())
        graph = self.graph_conv1.forward(graph, adj=adj1, relu=True)
        # print(graph.size(),x.size())
        graph = self.graph_2_fea.forward(graph, x)
        x = self.skip_conv(x)
        x = x + graph
        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def get_graph_parameter(self):
        l = []
        for name,k in self.named_parameters():
            if 'graph' in name:
                l.append(k)
        return l

    def get_semantic_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'semantic' in name:
                l.append(k)
        return l

class DeepLabv3_plus_symgcn_copy_5graphlayer(deeplab_xception.DeepLabv3_plus):
    def __init__(self,nInputChannels=3, n_classes=7, os=16,input_channels=256,hidden_layers=128,out_channels=256,adj1=None,adj2=None):
        super(DeepLabv3_plus_symgcn_copy_5graphlayer, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes, os=os)
        self.featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,hidden_layers=hidden_layers,
                                                           nodes=n_classes)
        self.graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.graph_conv4 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.graph_conv5 = gcn.GraphConvolution(hidden_layers, hidden_layers)

        self.graph_2_fea = gcn.Graph_to_Featuremaps(input_channels=input_channels,output_channels=out_channels,
                                                    hidden_layers=hidden_layers,nodes=n_classes
                                                    )
        self.skip_conv = nn.Sequential(*[nn.Conv2d(input_channels,input_channels,kernel_size=1),
                                         nn.ReLU(True)])
        self.semantic = nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
        # self.adj1 = adj1
        # self.adj2 = adj2\

    def forward(self, input, adj1=None, adj2=None):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)
        ### add graph
        # print('x size',x.size(),adj1.size())
        graph = self.featuremap_2_graph(x)
        # print(graph.size())
        graph = self.graph_conv1.forward(graph, adj=adj1, relu=True)
        graph = self.graph_conv2.forward(graph, adj=adj1, relu=True)
        graph = self.graph_conv3.forward(graph, adj=adj1, relu=True)
        graph = self.graph_conv4.forward(graph, adj=adj1, relu=True)
        graph = self.graph_conv5.forward(graph, adj=adj1, relu=True)
        # print(graph.size(),x.size())
        graph = self.graph_2_fea.forward(graph, x)
        x = self.skip_conv(x)
        x = x + graph
        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def get_graph_parameter(self):
        l = []
        for name,k in self.named_parameters():
            if 'graph' in name:
                l.append(k)
        return l

    def get_semantic_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'semantic' in name:
                l.append(k)
        return l

class DeepLabv3_plus_symgcn_copy_without_res(deeplab_xception.DeepLabv3_plus):
    def __init__(self,nInputChannels=3, n_classes=7, os=16,input_channels=256,hidden_layers=128,out_channels=256,adj1=None,adj2=None):
        super(DeepLabv3_plus_symgcn_copy_without_res, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes, os=os)
        self.featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,hidden_layers=hidden_layers,
                                                           nodes=n_classes)
        self.graph_conv1 = gcn.GraphConvolution(hidden_layers,hidden_layers)
        self.graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)

        self.graph_2_fea = gcn.Graph_to_Featuremaps(input_channels=input_channels,output_channels=out_channels,
                                                    hidden_layers=hidden_layers,nodes=n_classes
                                                    )
        self.skip_conv = nn.Sequential(*[nn.Conv2d(input_channels,input_channels,kernel_size=1),
                                         nn.ReLU(True)])
        self.semantic = nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
        # self.adj1 = adj1
        # self.adj2 = adj2\

    def forward(self, input, adj1=None, adj2=None):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)
        ### add graph
        # print('x size',x.size(),adj1.size())
        graph = self.featuremap_2_graph(x)
        # print(graph.size())
        graph = self.graph_conv1.forward(graph, adj=adj1, relu=True)
        graph = self.graph_conv2.forward(graph, adj=adj1, relu=True)
        graph = self.graph_conv3.forward(graph, adj=adj1, relu=True)
        # print(graph.size(),x.size())
        # graph = self.gcn_encode.forward(graph,relu=True)
        # graph = self.graph_conv2.forward(graph,adj=adj2,relu=True)
        # graph = self.gcn_decode.forward(graph,relu=True)
        graph = self.graph_2_fea.forward(graph,x)
        # x = self.skip_conv(x)
        x = graph
        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def get_graph_parameter(self):
        l = []
        for name,k in self.named_parameters():
            if 'graph' in name:
                l.append(k)
        return l

    def get_semantic_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'semantic' in name:
                l.append(k)
        return l


if __name__ == "__main__":
    # model = DeepLabv3_plus(nInputChannels=3, n_classes=21, os=16, pretrained=False, _print=True)
    # model.eval()
    # ckt = torch.load('C:\\Users\gaoyi\code_python\deeplab_v3plus.pth')
    # model.load_state_dict_new(ckt)
    from torch.autograd import Variable
    image = Variable(torch.randn(1, 3, 512, 512))
    # with torch.no_grad():
    #     output = model.forward(image)
    # print(output.size())
    # print(output)
    net = DeepLabv3_plus_gcn().cuda().eval()
    pic = torch.rand((1,3,64,64))
    adj1 = Variable(torch.from_numpy(pascal_adj).float())
    adj2 = Variable(torch.ones((2,2)))
    r = net.forward(image.cuda(),adj1.cuda(),adj2.cuda())
    '''gcn module test'''
    # A = torch.ones((7,7))
    # gcn_module = GCN_module(128,7,512,A)
    # a = torch.rand((2,128,7,7))
    # r = gcn_module.forward(a)
    # print(r.size())









