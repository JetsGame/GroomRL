# this file is adapted from dump_to_simple_cpp.py from keras2cpp

def check_model(hps):
    """Check that the model defined is portable to cpp."""
    if hps['dropout']>0.0 or hps['architecture']=='LSTM':
        raise ValueError("keras_to_cpp: Only Dense layers without Dropout are supported.")

def keras_to_cpp(model, layerdic, cpp_fn, verbose=False):
    """Convert keras to cpp readable file."""
    with open(cpp_fn, 'w') as fout:
        fout.write('layers ' + str(len(model.layers)) + '\n')
    
        layers = []
        for ind, l in enumerate(layerdic):
            if verbose:
                print(ind, l)
            fout.write('layer ' + str(ind) + ' ' + l['class_name'] + '\n')
    
            if verbose:
                print(str(ind), l['class_name'])
            layers += [l['class_name']]
            if l['class_name'] == 'Conv2D':
                #fout.write(str(l['config']['nb_filter']) + ' ' + str(l['config']['nb_col']) + ' ' + str(l['config']['nb_row']) + ' ')
    
                #if 'batch_input_shape' in l['config']:
                #    fout.write(str(l['config']['batch_input_shape'][1]) + ' ' + str(l['config']['batch_input_shape'][2]) + ' ' + str(l['config']['batch_input_shape'][3]))
                #fout.write('\n')
    
                W = model.layers[ind].get_weights()[0]
                if verbose:
                    print(W.shape)
                fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + ' ' + str(W.shape[2]) + ' ' + str(W.shape[3]) + ' ' + l['config']['padding'] + '\n')
    
                for i in range(W.shape[0]):
                    for j in range(W.shape[1]):
                        for k in range(W.shape[2]):
                            fout.write(str(W[i,j,k]) + '\n')
                fout.write(str(model.layers[ind].get_weights()[1]) + '\n')
    
            if l['class_name'] == 'Activation':
                fout.write(l['config']['activation'] + '\n')
            if l['class_name'] == 'MaxPooling2D':
                fout.write(str(l['config']['pool_size'][0]) + ' ' + str(l['config']['pool_size'][1]) + '\n')
            #if l['class_name'] == 'Flatten':
            #    print(l['config']['name'])
            if l['class_name'] == 'Dense':
                #fout.write(str(l['config']['output_dim']) + '\n')
                W = model.layers[ind].get_weights()[0]
                if verbose:
                    print(W.shape)
                fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + '\n')
    
                for w in W:
                    fout.write(str(w) + '\n')
                fout.write(str(model.layers[ind].get_weights()[1]) + '\n')
