\begindata

         PATH_SYMBOLS += ('IMAP', 'TESTS', 'TOOLS')
         PATH_VALUES  += (
            '/Users/nake7532/Projects/IMAP/imap_processing/',
            'imap_processing/tests/spice/test_data/',
            'tools/tests/test_data/spice/' )


      KERNELS_TO_LOAD += (
       '$IMAP/$TESTS/imap_sclk_0000.tsc'
       '$IMAP/$TESTS/naif0012.tls'
       '$IMAP/$TESTS/imap_science_0001.tf'
       '$IMAP/$TESTS/sim_1yr_imap_pointing_frame.bc'
      )
\begintext