import numpy as np

def check_vector_data(varname, vdata, mpi, nl_colm, spv_in=None, largevalue=None):
    if mpi.p_is_worker:
        if spv_in is not None:
            spv = spv_in
        else:
            spv = -1.0e36

        if np.any(vdata != spv):
            vmin = np.min(vdata[vdata != spv])
            vmax = np.max(vdata[vdata != spv])
        else:
            vmin = spv
            vmax = spv

        has_nan = np.any(np.isnan(vdata))

        if nl_colm['USEMPI']:
            pass

        if mpi.p_iam_worker == mpi.p_root:
            info = ''

            if has_nan:
                info += ' with NAN'

            if largevalue is not None:
                if max(abs(vmin), abs(vmax)) > largevalue:
                    ss = f'{largevalue:.2e}'
                    info += f' with value > {ss}'

            wfmt = '(Check vector data:'+ varname+ ' is in ('+ vmin+ ','+ vmax+ ')'+ info+')'
            print(wfmt.format(varname, vmin, vmax, info))

            # Assuming CoLMDEBUG is defined elsewhere
            if nl_colm['CoLMDEBUG']:
                if len(info.strip()) > 0:
                    mpi.CoLM_stop()

