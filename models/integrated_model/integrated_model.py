# Image
#   |
# Green/Grey Discriminator -Green/Grey IR-|
#   |                                     |
# IR Discriminator --Red IR Image---------|
#   |                                     |
#  RGB                                ToGrayscale
#   |                                     |
#   |                                     |
# RGB Backbone                       IR Backbone
#   |                                     |
# loss                                   loss
#   |                                     |
# Backward                            Backward
